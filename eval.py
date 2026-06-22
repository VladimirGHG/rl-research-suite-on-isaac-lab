"""
eval.py — Evaluation entry point.

Loads a trained policy (local checkpoint or HuggingFace Hub) and runs
deterministic evaluation episodes.
"""

import os
import hydra
import torch
import numpy as np
import wandb
from omegaconf import DictConfig

from envs.env import IsaacLabPlatformEnv
from algorithms.sac import SACTrainer
from algorithms.ppo import PPOTrainer
from policies.from_hub import SB3HubPolicyWrapper


def _load_local_policy(checkpoint_path: str, env, algo_cfg):
    ckpt = torch.load(checkpoint_path, map_location=env.device)
    if "qf1" in ckpt:
        trainer = SACTrainer(env, algo_cfg)
    elif "critic" in ckpt:
        trainer = PPOTrainer(env, algo_cfg)
    else:
        raise ValueError(f"Unrecognized checkpoint format: {list(ckpt.keys())}")
    trainer.load(checkpoint_path)
    return trainer.to_policy()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    cfg.env.num_envs = 1
    env = IsaacLabPlatformEnv(cfg=cfg.env)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"eval_{cfg.algo.get('name', 'policy')}_{cfg.env.get('task_name', '')}",
        mode=cfg.wandb.mode,
    )

    local_ckpt = cfg.get("checkpoint_path", None)
    if local_ckpt and os.path.exists(local_ckpt):
        policy = _load_local_policy(local_ckpt, env, cfg.algo)
    else:
        policy = SB3HubPolicyWrapper(
            repo_id=cfg.experiment.eval_repo,
            filename=cfg.experiment.eval_file,
            env=env,
        )

    policy.env_policy_actions_compatibility(env)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    ep_reward, ep_length = 0.0, 0
    all_returns, all_lengths, success_count = [], [], 0
    frames = []
    capture_video = cfg.get("video_dump", False)

    with torch.no_grad():
        for _ in range(cfg.experiment.eval_steps):
            action, _ = policy.predict(obs, deterministic=True)
            obs_dict, reward, terminated, truncated, _ = env.step(action)
            obs = obs_dict["policy"]
            ep_reward += reward.mean().item()
            ep_length += 1

            if capture_video:
                pixels = env._get_synthetic_pixels()
                frames.append((pixels[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

            if terminated.any() or truncated.any():
                all_returns.append(ep_reward)
                all_lengths.append(ep_length)
                if terminated.any().item():
                    success_count += 1
                ep_reward, ep_length = 0.0, 0
                obs_dict, _ = env.reset()
                obs = obs_dict["policy"]

    if all_returns:
        run.log({
            "eval/mean_return": float(np.mean(all_returns)),
            "eval/std_return": float(np.std(all_returns)),
            "eval/success_rate": success_count / len(all_returns),
            "eval/mean_length": float(np.mean(all_lengths)),
            "eval/num_episodes": len(all_returns),
        })
        if capture_video and frames:
            video = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
            run.log({"eval/video": wandb.Video(video, fps=30, format="mp4")})

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
