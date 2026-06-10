"""
eval.py — Evaluation entry point.

Loads a trained policy (local checkpoint or HuggingFace Hub) and runs
deterministic evaluation episodes.

FIX: Added success rate metric, W&B logging, and video capture — all three
were missing from the original but required by spec §5 Week 5.
"""

import os
import hydra
import torch
import numpy as np
import wandb
from omegaconf import DictConfig

from envs.env import IsaacLabPlatformEnv
from policies.from_hub import SB3HubPolicyWrapper


def _load_local_policy(checkpoint_path: str, env):
    """
    Load a locally saved algorithm checkpoint.
    Supports PPO, SAC — detects by trying each trainer.
    """
    device = env.device
    ckpt   = torch.load(checkpoint_path, map_location=device)

    # Detect algo from checkpoint keys
    if "actor" in ckpt and "qf1" in ckpt:
        # SAC checkpoint
        from algorithms.sac import SACActorNetwork
        action_low  = torch.tensor(env.action_space.low,  device=device)
        action_high = torch.tensor(env.action_space.high, device=device)
        actor = SACActorNetwork(512, env.action_space.shape[0], action_low, action_high).to(device)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()

        from algorithms.sac import _FrozenEncoder
        encoder = _FrozenEncoder().to(device)

        class _SACPolicy:
            def predict(self, obs, deterministic=True):
                pixels = obs["policy"] if isinstance(obs, dict) else obs
                with torch.no_grad():
                    features = encoder(pixels.float())
                    _, _, mean = actor.get_action(features)
                return mean, None

        return _SACPolicy()

    elif "actor" in ckpt and "critic" in ckpt:
        # PPO checkpoint
        from algorithms.ppo import PPOActor, _FrozenEncoder
        actor   = PPOActor(512, env.action_space.shape[0]).to(device)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        encoder = _FrozenEncoder().to(device)

        class _PPOPolicy:
            def predict(self, obs, deterministic=True):
                pixels = obs["policy"] if isinstance(obs, dict) else obs
                with torch.no_grad():
                    features = encoder(pixels.float())
                    return actor.actor_mean(features), None

        return _PPOPolicy()

    raise ValueError(f"Cannot detect algorithm from checkpoint keys: {list(ckpt.keys())}")


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Single env for deterministic evaluation
    cfg.env.num_envs = 1

    print(f"[eval] Initialising environment: {cfg.env.get('task_name', 'unknown')}")
    env = IsaacLabPlatformEnv(cfg=cfg.env)

    # ── W&B init ──────────────────────────────────────────────────────────
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"eval_{cfg.algo.get('name','policy')}_{cfg.env.get('task_name','')}",
        mode=cfg.wandb.mode,
        config={"eval_steps": cfg.experiment.eval_steps},
    )

    # ── Load policy ───────────────────────────────────────────────────────
    local_ckpt = cfg.get("checkpoint_path", None)
    if local_ckpt and os.path.exists(local_ckpt):
        print(f"[eval] Loading local checkpoint: {local_ckpt}")
        policy = _load_local_policy(local_ckpt, env)
    else:
        print(f"[eval] Loading HF Hub policy: {cfg.experiment.eval_repo}")
        policy = SB3HubPolicyWrapper(
            repo_id =cfg.experiment.eval_repo,
            filename=cfg.experiment.eval_file,
            env     =env,
        )

    # ── Evaluation loop ───────────────────────────────────────────────────
    obs, _   = env.reset()
    ep_reward  = 0.0
    ep_length  = 0
    all_returns  = []
    all_lengths  = []
    success_count= 0

    # Video capture setup
    frames = []
    capture_video = cfg.get("video_dump", False)

    print(f"[eval] Running {cfg.experiment.eval_steps} steps...")

    with torch.no_grad():
        for step in range(cfg.experiment.eval_steps):

            action, _ = policy.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, extras = env.step(action)

            ep_reward += reward.detach().cpu().numpy().mean()
            ep_length += 1

            # Video frame capture
            if capture_video:
                try:
                    pixels = env._get_synthetic_pixels()
                    frame  = (pixels[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    frames.append(frame)
                except Exception:
                    pass

            # Episode end
            if terminated.any() or truncated.any():
                all_returns.append(ep_reward)
                all_lengths.append(ep_length)

                # Success = episode ended via task completion (not timeout)
                is_success = bool(terminated.any().item())
                if is_success:
                    success_count += 1

                print(
                    f"[eval] Episode {len(all_returns):3d} | "
                    f"return={ep_reward:.3f} | "
                    f"length={ep_length} | "
                    f"success={'YES' if is_success else 'no'}"
                )

                ep_reward = 0.0
                ep_length = 0
                obs, _    = env.reset()

    # ── Summary metrics ───────────────────────────────────────────────────
    if all_returns:
        mean_return  = float(np.mean(all_returns))
        std_return   = float(np.std(all_returns))
        success_rate = success_count / len(all_returns)

        print(f"\n[eval] === Results over {len(all_returns)} episodes ===")
        print(f"  Mean return:  {mean_return:.3f} ± {std_return:.3f}")
        print(f"  Success rate: {success_rate*100:.1f}%")
        print(f"  Mean length:  {float(np.mean(all_lengths)):.1f} steps")

        # Log to W&B
        run.log({
            "eval/mean_return":  mean_return,
            "eval/std_return":   std_return,
            "eval/success_rate": success_rate,
            "eval/mean_length":  float(np.mean(all_lengths)),
            "eval/num_episodes": len(all_returns),
        })

        # Video logging to W&B
        if capture_video and len(frames) > 0:
            # frames: list of (H, W, 3) uint8 → W&B video expects (T, H, W, C)
            video_array = np.stack(frames, axis=0)
            run.log({
                "eval/video": wandb.Video(
                    video_array.transpose(0, 3, 1, 2),  # (T, C, H, W)
                    fps=30, format="mp4"
                )
            })
            print(f"[eval] Video logged to W&B ({len(frames)} frames)")
    else:
        print("[eval] No complete episodes recorded.")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
