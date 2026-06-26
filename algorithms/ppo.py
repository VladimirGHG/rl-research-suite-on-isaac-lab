import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.base import Trainer
from encoders.resnet18 import FrozenResNet18
from policies.custom.ppo_policy import PPOActorNetwork, PPOCriticNetwork

class _RolloutBuffer:
    """Temporary storage for on-policy rollouts, kept entirely on GPU."""

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int, device: str):
        self.max_size = num_steps
        self.num_envs = num_envs
        self.device = device
        self.ptr = 0

        self.obs      = torch.zeros((num_steps, num_envs, obs_dim),    device=device)
        self.actions  = torch.zeros((num_steps, num_envs, action_dim), device=device)
        self.logprobs = torch.zeros((num_steps, num_envs),             device=device)
        self.rewards  = torch.zeros((num_steps, num_envs),             device=device)
        self.dones    = torch.zeros((num_steps, num_envs),             device=device)
        self.values   = torch.zeros((num_steps, num_envs),             device=device)

    def add(self, obs, action, logprob, reward, done, value):
        self.obs[self.ptr]      = obs.detach()
        self.actions[self.ptr]  = action.detach()
        self.logprobs[self.ptr] = logprob.detach()
        self.rewards[self.ptr]  = reward.detach()
        self.dones[self.ptr]    = done.detach()
        self.values[self.ptr]   = value.detach()
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.max_size

    def clear(self):
        self.ptr = 0

class PPOTrainer(Trainer):
    """Proximal Policy Optimization (PPO) trainer for Isaac Lab."""

    def __init__(self, env, algo_cfg, wandb_run=None, resume_path=None):
        super().__init__(env, algo_cfg, wandb_run, resume_path)
        self.device = env.device

        self.enc_dim    = 512
        self.action_dim = env.action_space.shape[0]
        self.num_envs   = getattr(env, "num_envs", 1)
        self.num_steps  = int(algo_cfg.num_steps)

        self.action_low  = torch.tensor(env.action_space.low,  device=self.device)
        self.action_high = torch.tensor(env.action_space.high, device=self.device)

        self.encoder = FrozenResNet18().to(self.device)
        self.actor   = PPOActorNetwork(self.enc_dim, self.action_dim).to(self.device)
        self.critic  = PPOCriticNetwork(self.enc_dim).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=algo_cfg.learning_rate, eps=1e-5,
        )

        self.rb = _RolloutBuffer(
            self.num_steps, self.num_envs, self.enc_dim, self.action_dim, self.device
        )

        self.global_step = 0
        self.current_obs = None
        self.next_done   = None
        self.start_time  = None

        if self.resume_path and os.path.exists(self.resume_path):
            self.load(self.resume_path)

    def _encode(self, obs_dict) -> torch.Tensor:
        """Extract visual features; handles both dict and raw tensor obs."""
        pixels = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        return self.encoder(pixels.float())

    def _init_rollout(self):
        """Reset environment and initialise carry-over state."""
        obs_dict, _ = self.env.reset()
        self.current_obs = self._encode(obs_dict)
        self.next_done   = torch.zeros(self.num_envs, device=self.device)

    def collect_rollout(self) -> dict:
        """
        Collect exactly `num_steps` environment steps across all envs.
        The previous version only collected ONE step per call, so
        `update()` always saw an incomplete buffer and returned early.
        """
        if self.current_obs is None:
            self._init_rollout()

        self.rb.clear() # always start fresh

        self.actor.train()
        self.critic.train()

        for _ in range(self.num_steps):
            with torch.no_grad():
                action, logprob, _ = self.actor.get_action(self.current_obs)
                value = self.critic(self.current_obs).flatten()

            clamped_action = torch.clamp(action, self.action_low, self.action_high)
            next_obs_dict, rewards, terminated, truncated, _ = self.env.step(clamped_action)

            done = (terminated | truncated).float()
            self.rb.add(
                self.current_obs,
                action,
                logprob,
                rewards.float(),
                done,
                value,
            )

            self.current_obs = self._encode(next_obs_dict)
            self.next_done   = done
            self.global_step += self.num_envs

        return {"global_step": self.global_step}

    def update(self) -> dict:
        """Compute GAE and run PPO clipped-objective epochs."""
        cfg = self.config

        with torch.no_grad():
            next_value = self.critic(self.current_obs).flatten()
            advantages  = torch.zeros_like(self.rb.rewards)
            lastgaelam  = 0

            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues      = next_value
                else:
                    nextnonterminal = 1.0 - self.rb.dones[t + 1]
                    nextvalues      = self.rb.values[t + 1]

                delta          = self.rb.rewards[t] + cfg.gamma * nextvalues * nextnonterminal - self.rb.values[t]
                advantages[t]  = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + self.rb.values

        b_obs        = self.rb.obs.reshape(-1, self.enc_dim)
        b_actions    = self.rb.actions.reshape(-1, self.action_dim)
        b_logprobs   = self.rb.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)
        b_values     = self.rb.values.reshape(-1)

        batch_size    = self.num_steps * self.num_envs
        minibatch_size = batch_size // int(cfg.num_minibatches)
        b_inds = np.arange(batch_size)

        pg_loss_acc, v_loss_acc, ent_loss_acc = 0.0, 0.0, 0.0
        num_updates = int(cfg.update_epochs) * int(cfg.num_minibatches)

        for _ in range(int(cfg.update_epochs)):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start : start + minibatch_size]

                _, newlogprob, entropy = self.actor.get_action(
                    b_obs[mb_inds], action=b_actions[mb_inds]
                )
                newvalue = self.critic(b_obs[mb_inds]).flatten()

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio    = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef),
                ).mean()

                # Value loss
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss = 0.5 * torch.max(
                        v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2
                    ).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                pg_loss_acc  += pg_loss.item()
                v_loss_acc   += v_loss.item()
                ent_loss_acc += entropy_loss.item()

        return {
            "losses/policy_loss":  pg_loss_acc  / num_updates,
            "losses/value_loss":   v_loss_acc   / num_updates,
            "losses/entropy_loss": ent_loss_acc / num_updates,
        }

    def train(self):
        """Main training loop for PPO."""
        cfg = self.config
        total_timesteps = int(cfg.total_timesteps)
        ckpt_every      = int(cfg.checkpoint_interval)
        
        eval_every    = int(getattr(cfg, "eval_interval", 100000))
        eval_episodes = int(getattr(cfg, "eval_episodes", 5))
        
        self.start_time = time.time()

        steps_per_update = self.num_steps * self.num_envs
        expected_updates = total_timesteps // steps_per_update

        print(
            f"[PPO] Starting | total_steps={total_timesteps:,} "
            f"| num_envs={self.num_envs} | num_steps={self.num_steps} "
            f"| steps_per_update={steps_per_update:,} "
            f"| expected_updates={expected_updates:,}"
        )

        update_count = 0
        last_ckpt_step = self.global_step
        last_eval_step = self.global_step # Tracker for evaluation

        while self.global_step < total_timesteps:
            self.collect_rollout()
            metrics = self.update()
            update_count += 1

            elapsed = time.time() - self.start_time
            sps = int(self.global_step / elapsed) if elapsed > 0 else 0

            metrics["charts/global_step"] = self.global_step
            metrics["charts/SPS"]         = sps
            metrics["charts/update"]      = update_count

            # Log to terminal
            print(
                f"[PPO] update={update_count} step={self.global_step:,} "
                f"| SPS={sps} "
                f"| v_loss={metrics['losses/value_loss']:.4f} "
                f"| p_loss={metrics['losses/policy_loss']:.4f}"
            )

            # Log to W&B every update
            if self.wandb_run is not None:
                self.wandb_run.log(metrics, step=self.global_step)

            # Evaluation: trigger once per eval_every steps
            if self.global_step - last_eval_step >= eval_every:
                eval_metrics = self.evaluate(eval_episodes)
                print(
                    f"[PPO] eval @ step={self.global_step:,} | "
                    f"mean_return={eval_metrics['eval/mean_return']:.3f} | "
                    f"std_return={eval_metrics['eval/std_return']:.3f}"
                )
                if self.wandb_run is not None:
                    # Log the eval metrics alongside the global step
                    eval_metrics["charts/global_step"] = self.global_step
                    self.wandb_run.log(eval_metrics, step=self.global_step)
                
                last_eval_step = self.global_step

            # Checkpoint: trigger once per ckpt_every steps
            if self.global_step - last_ckpt_step >= ckpt_every:
                self.save(f"checkpoints/ppo_step_{self.global_step}.pt")
                last_ckpt_step = self.global_step

        print(f"[PPO] Training complete — {self.global_step:,} steps over {update_count} updates.")
 
    def evaluate(self, num_of_episodes: int) -> dict:
        """Deterministic evaluation using policy mean."""
        self.actor.eval()
        self.critic.eval()

        returns   = []
        obs_dict, _ = self.env.reset()
        obs       = self._encode(obs_dict)
        ep_return = torch.zeros(self.num_envs, device=self.device)

        with torch.no_grad():
            while len(returns) < num_of_episodes:
                # actor.forward() returns the mean — correct for deterministic eval
                action = self.actor(obs)
                clamped_action = torch.clamp(action, self.action_low, self.action_high)

                obs_dict, reward, terminated, truncated, _ = self.env.step(clamped_action)
                obs = self._encode(obs_dict)

                ep_return += reward.float()
                done = terminated | truncated

                for i, d in enumerate(done):
                    if d:
                        returns.append(ep_return[i].item())
                        ep_return[i] = 0.0

        self.actor.train()
        self.critic.train()

        return {
            "eval/mean_return": float(np.mean(returns[:num_of_episodes])),
            "eval/std_return":  float(np.std(returns[:num_of_episodes])),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "global_step": self.global_step,
                "actor":       self.actor.state_dict(),
                "critic":      self.critic.state_dict(),
                "optimizer":   self.optimizer.state_dict(),
            },
            path,
        )
        print(f"[PPO] Saved -> {path} (step {self.global_step})")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.global_step = ckpt["global_step"]
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"[PPO] Loaded from {path} (step {self.global_step})")