import copy
import os

import torch
import torch.nn.functional as F

import copy
import os

import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import Trainer
from policies.custom.td3_policy import TD3Policy
from envs.managers.common import get_ee_position

# The PixelReplayBuffer class manages the storage and sampling of experiences for training.
class PixelReplayBuffer(object):
    def __init__(self, num_envs: int, max_size: int, obs_dim: int, action_dim: int, device):
        self.max_size = max_size
        self.num_envs = num_envs
        self.device = device # GPU is used only at sample time to move batches
        self.ptr = 0
        self.size = 0

        cpu = torch.device("cpu")
        self.states = torch.zeros((max_size, num_envs, obs_dim),    dtype=torch.float32, device=cpu)
        self.actions = torch.zeros((max_size, num_envs, action_dim), dtype=torch.float32, device=cpu)
        self.next_states = torch.zeros((max_size, num_envs, obs_dim),    dtype=torch.float32, device=cpu)
        self.rewards = torch.zeros((max_size, num_envs, 1), dtype=torch.float32, device=cpu)
        self.not_dones = torch.zeros((max_size, num_envs, 1), dtype=torch.float32, device=cpu)

    # The add method stores a new experience in the buffer, updating the current pointer and size accordingly.
    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state.detach().cpu()
        self.actions[self.ptr] = action.detach().cpu()
        self.next_states[self.ptr] = next_state.detach().cpu()
        self.rewards[self.ptr] = reward.unsqueeze(-1).detach().cpu()
        self.not_dones[self.ptr] = (~done).unsqueeze(-1).float().detach().cpu()

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # The sample method retrieves a batch of experiences from the buffer, randomly selecting indices for both time steps and environments,
    # and returns the corresponding states, actions, next states, rewards, and not-done flags.
    def sample(self, batch_size: int):
        # Each returned tensor is moved to self.device (GPU) for the update step.
        step_idx = torch.randint(0, self.size, (batch_size,))
        env_idx  = torch.randint(0, self.num_envs, (batch_size,))
        return (
            self.states[step_idx, env_idx].to(self.device),
            self.actions[step_idx, env_idx].to(self.device),
            self.next_states[step_idx, env_idx].to(self.device),
            self.rewards[step_idx, env_idx].to(self.device),
            self.not_dones[step_idx, env_idx].to(self.device),
        )


# The main policy class, based on processing pixel observations through an encoder and then using the actor network to select actions.
class Td3Trainer(Trainer):
    def __init__(self, env, policy: TD3Policy, algo_cfg, wandb_run=None, resume_path=None):
        self.policy: TD3Policy = policy
        super().__init__(env=env, algo_cfg=algo_cfg, wandb_run=wandb_run, resume_path=resume_path)

        self.device = env.device

        self.discount = algo_cfg.get("discount", 0.99)
        self.tau = algo_cfg.get("tau", 0.005)
        self.policy_noise = algo_cfg.get("policy_noise", 0.2)
        self.noise_clip = algo_cfg.get("noise_clip", 0.5)
        self.policy_freq = algo_cfg.get("policy_freq", 2)
        self.batch_size = algo_cfg.get("batch_size", 256)
        self.expl_noise = algo_cfg.get("expl_noise", 0.1)
        self.learning_starts = algo_cfg.get("learning_starts", 5000)
        self.total_timesteps = algo_cfg.get("total_timesteps", 1000000)
        self.checkpoint_interval = algo_cfg.get("checkpoint_interval", 20000)

        self.action_dim = env.action_space.shape[0]

        self.num_envs = getattr(env, "num_envs", 1)
        self.total_it = 0
        self.total_steps_collected = 0

        self.actor_target = copy.deepcopy(self.policy.actor)
        self.critic_target = copy.deepcopy(self.policy.critic)

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=algo_cfg.get("lr_actor", 3e-4))
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=algo_cfg.get("lr_critic", 3e-4))

        self.replay_buffer = PixelReplayBuffer(
            num_envs=self.num_envs,
            max_size=algo_cfg.get("buffer_capacity", 50000),
            obs_dim=self.policy.encoder.feature_dim + 9, # 512 visual + 9 state (obj_pos + ee_pos + rel_pos)
            action_dim=self.action_dim,
            device=self.device
        )
        self.current_obs, _ = self.env.reset()

        if resume_path and os.path.exists(resume_path):
            self.load(resume_path)
            print(f"[TD3] Resumed from: {resume_path} (it {self.total_it})")

    def collect_rollout(self) -> dict:
        # Get raw pixels and encode them
        raw = self.env.scene["franka_wrist_camera"].data.output["rgb"]

        raw_pixels = raw.permute(0, 3, 1, 2).float() / 255.0
        
        with torch.no_grad():
            visual_features = self.policy.encoder(raw_pixels)
            
            # Get state features
            object_pos = self.env.scene["object"].data.root_pos_w
            ee_pos = get_ee_position(self.env, "robot")
            relative_pos = object_pos - ee_pos
            
            # Normalize by dividing by 2.0
            object_pos_norm = object_pos / 2.0
            ee_pos_norm = ee_pos / 2.0
            relative_pos_norm = relative_pos / 2.0
            
            state_features = torch.cat([
                visual_features,
                object_pos_norm,
                ee_pos_norm,
                relative_pos_norm
            ], dim=-1)

        if self.total_steps_collected < self.learning_starts:
            actions = torch.rand((self.num_envs, self.action_dim), device=self.device) * 2.0 - 1.0
            actions = actions * self.policy.max_action
        else:
            actions = self.policy.get_action(state_features)
            noise = torch.randn_like(actions) * self.expl_noise * 1.5
            actions = (actions + noise).clamp(-self.policy.max_action, self.policy.max_action)

        next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = terminated | truncated

        with torch.no_grad():
            next_raw = self.env.scene["franka_wrist_camera"].data.output["rgb"]
            next_pixels = next_raw.permute(0, 3, 1, 2).float() / 255.0
            next_visual = self.policy.encoder(next_pixels)
            
            next_object_pos = self.env.scene["object"].data.root_pos_w
            next_ee_pos = get_ee_position(self.env, "robot")
            next_relative_pos = next_object_pos - next_ee_pos
            
            next_object_pos_norm = next_object_pos / 2.0
            next_ee_pos_norm = next_ee_pos / 2.0
            next_relative_pos_norm = next_relative_pos / 2.0
            
            next_state_features = torch.cat([
                next_visual,
                next_object_pos_norm,
                next_ee_pos_norm,
                next_relative_pos_norm
            ], dim=-1)

        self.replay_buffer.add(
            state_features,
            actions,
            next_state_features,
            rewards.to(self.device),
            dones.to(self.device)
        )

        self.current_obs = next_obs
        self.total_steps_collected += self.num_envs

        return {
            "mean_rollout_reward": rewards.mean().item(),
            "buffer_current_size": self.replay_buffer.size * self.num_envs,
            "total_steps_collected": self.total_steps_collected
        }

    def update(self) -> dict:
        """Performs a single update step for the TD3 algorithm, including critic and actor updates, and returns relevant metrics."""

        if self.total_steps_collected < self.learning_starts or \
        self.replay_buffer.size * self.num_envs < self.batch_size:
            return {"status": "Warm-up phase; skipping gradient updates."}

        self.total_it += 1
        metrics = {}
        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.policy.max_action, self.policy.max_action
            )
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.policy.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics["critic_loss"] = critic_loss.item()
        metrics["mean_q_estimation"] = current_Q1.mean().item()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.policy.critic.Q1(state, self.policy.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            metrics["actor_loss"] = actor_loss.item()

            for param, target_param in zip(self.policy.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.policy.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return metrics

    def evaluate(self) -> dict:
        """Runs deterministic evaluation episodes and returns the mean reward across all environments."""

        obs, _ = self.env.reset()
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        while not done.all():
            with torch.no_grad():
                raw = self.env.scene["franka_wrist_camera"].data.output["rgb"]
                pixels = raw.permute(0, 3, 1, 2).float() / 255.0
                visual_features = self.policy.encoder(pixels)
                
                object_pos = self.env.scene["object"].data.root_pos_w
                ee_pos = get_ee_position(self.env, "robot")
                relative_pos = object_pos - ee_pos
                
                object_pos_norm = object_pos / 2.0
                ee_pos_norm = ee_pos / 2.0
                relative_pos_norm = relative_pos / 2.0
                
                features = torch.cat([
                    visual_features,
                    object_pos_norm,
                    ee_pos_norm,
                    relative_pos_norm
                ], dim=-1)
                
                actions = self.policy.get_action(features)

            next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
            episode_rewards += rewards.to(self.device) * (~done)
            done |= (terminated | truncated)
            obs = next_obs

        return {"eval_mean_reward": float(episode_rewards.mean().item())}

    def save(self, path: str):
        """
        Saves the current state of the trainer, including the policy networks, 
        target networks, optimizers, and training progress to a specified file path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "actor_state": self.policy.actor.state_dict(),
            "critic_state": self.policy.critic.state_dict(),
            "actor_target_state": self.actor_target.state_dict(),
            "critic_target_state": self.critic_target.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
            "total_steps_collected": self.total_steps_collected
        }
        torch.save(save_dict, path)

    def load(self, path: str):
        """Loads the trainer state from a specified file path."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.actor.load_state_dict(checkpoint["actor_state"])
        self.policy.critic.load_state_dict(checkpoint["critic_state"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        self.total_it = checkpoint["total_it"]
        self.total_steps_collected = checkpoint.get("total_steps_collected", 0)

    def train(self):
        ckpt_every = int(self.checkpoint_interval)
        eval_every = int(self.config.get("eval_interval", 50000))
        eval_episodes = int(self.config.get("eval_episodes", 5))

        print(f"[TD3] Starting | total_steps={self.total_timesteps} | "
            f"num_envs={self.num_envs} | learning_starts={self.learning_starts}")

        while self.total_steps_collected < self.total_timesteps:
            prev_steps = self.total_steps_collected
            rollout_metrics = self.collect_rollout()
            update_metrics = self.update()

            if self.wandb_run is not None and \
            prev_steps // 100 != self.total_steps_collected // 100:
                log_payload = {**rollout_metrics, **update_metrics}
                log_payload["charts/global_step"] = self.total_steps_collected
                self.wandb_run.log(log_payload, step=self.total_steps_collected)

            if prev_steps // eval_every != self.total_steps_collected // eval_every:
                saved_obs = {k: v.clone() for k, v in self.current_obs.items()}

                eval_metrics = self.evaluate()
                print(f"[TD3] eval @ step={self.total_steps_collected} | {eval_metrics}")
                if self.wandb_run is not None:
                    self.wandb_run.log(eval_metrics, step=self.total_steps_collected)

                self.current_obs = saved_obs

            if prev_steps // ckpt_every != self.total_steps_collected // ckpt_every:
                self.save(f"checkpoints/td3_step_{self.total_steps_collected}.pt")

        print(f"[TD3] Training complete — {self.total_steps_collected} steps.")

