"""
PPO Trainer — adapted from CleanRL's ppo_continuous_action.py for Isaac Lab.

Owner: Student A (ppo-track)

Key differences from SAC:
  - On-Policy: Uses a temporary RolloutBuffer, not a massive ReplayBuffer.
  - Architecture: Shared Encoder -> Separate Actor (Stochastic) & Critic (Value).
  - Stability: Uses GAE (Generalized Advantage Estimation) and Clipped Objective.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torchvision import models

from algorithms.base import Trainer # Assuming this is your base class

# =============================================================================
# Standalone frozen ResNet18 encoder 
# =============================================================================
class StandaloneFrozenResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.features.parameters():
            param.requires_grad = False
        self.out_features = resnet.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.features(x)
            return x.view(x.size(0), -1)

# =============================================================================
# PPO Agent (Actor & Critic)
# =============================================================================
class PPOAgent(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.encoder = StandaloneFrozenResNet18()
        
        # Critic: Guesses the Value of the state
        self.critic = nn.Sequential(
            nn.Linear(self.encoder.out_features, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Actor: Guesses the best action (Mean)
        self.actor_mean = nn.Sequential(
            nn.Linear(self.encoder.out_features, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        
        # Actor: Standard Deviation (Uncertainty/Exploration)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        features = self.encoder(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.encoder(x)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Create Gaussian Distribution
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(features)

# =============================================================================
# Temporary Rollout Buffer
# =============================================================================
class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, action_dim, device):
        self.obs = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
        self.actions = torch.zeros((num_steps, num_envs, action_dim)).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)
        self.step = 0

    def add(self, obs, action, logprob, reward, done, value):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value.flatten()
        self.step += 1

    def reset(self):
        self.step = 0

# =============================================================================
# The Trainer Class
# =============================================================================
class PPOTrainer(Trainer):
    def __init__(self, env, algo_cfg, wandb_run=None, resume_path=None):
        super().__init__(env, algo_cfg, wandb_run, resume_path)
        
        self.num_envs = self.env.unwrapped.num_envs
        self.action_dim = self.env.action_space.shape[0]
        # Assuming TiledCamera output shape, adjust if different
        self.obs_shape = self.env.observation_space.shape 
        
        self.agent = PPOAgent(self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.algo_cfg.lr, eps=1e-5)
        
        self.buffer = RolloutBuffer(
            self.algo_cfg.num_steps, self.num_envs, self.obs_shape, self.action_dim, self.device
        )

        if resume_path:
            self.load(resume_path)

    def collect_rollout(self):
        """Play the simulation to collect N steps of data."""
        self.buffer.reset()
        obs, _ = self.env.reset()
        
        for step in range(self.algo_cfg.num_steps):
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(obs)
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = torch.logical_or(terminated, truncated).float()
            
            self.buffer.add(obs, action, logprob, reward, done, value)
            obs = next_obs
            self.global_step += self.num_envs
            
        return obs # Return final obs to compute final advantage

    def compute_advantages(self, next_obs):
        """Calculate GAE (How much better were the actions than expected?)"""
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(self.buffer.rewards).to(self.device)
            lastgaelam = 0
            
            for t in reversed(range(self.algo_cfg.num_steps)):
                if t == self.algo_cfg.num_steps - 1:
                    nextnonterminal = 1.0 - 0.0 # Assuming infinite horizon for simplicity here, adjust if needed
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.buffer.dones[t + 1]
                    nextvalues = self.buffer.values[t + 1]
                    
                delta = self.buffer.rewards[t] + self.algo_cfg.gamma * nextvalues * nextnonterminal - self.buffer.values[t]
                advantages[t] = lastgaelam = delta + self.algo_cfg.gamma * self.algo_cfg.gae_lambda * nextnonterminal * lastgaelam
                
            returns = advantages + self.buffer.values
            return advantages, returns

    def update(self, advantages, returns):
        """The Clipped Objective Update"""
        # Flatten the batch
        b_obs = self.buffer.obs.view((-1,) + self.obs_shape)
        b_logprobs = self.buffer.logprobs.view(-1)
        b_actions = self.buffer.actions.view((-1, self.action_dim))
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = self.buffer.values.view(-1)

        # Normalize advantages (Standard practice for stability)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Iterate over the batch data
        batch_size = self.algo_cfg.num_steps * self.num_envs
        b_inds = torch.arange(batch_size)
        
        for epoch in range(self.algo_cfg.update_epochs):
            # Mini-batching would go here. For simplicity, computing on full batch.
            # In a full implementation, you shuffle b_inds and split into mini_batches
            
            _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs, b_actions)
            logratio = newlogprob - b_logprobs
            ratio = logratio.exp()

            # PPO Clipped Objective Math
            mb_advantages = b_advantages
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.algo_cfg.clip_coef, 1 + self.algo_cfg.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value Loss
            v_loss = 0.5 * ((newvalue.view(-1) - b_returns) ** 2).mean()

            # Entropy Bonus (Encourage exploration)
            entropy_loss = entropy.mean()
            
            # Total Loss
            loss = pg_loss - self.algo_cfg.ent_coef * entropy_loss + v_loss * self.algo_cfg.vf_coef

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.algo_cfg.max_grad_norm)
            self.optimizer.step()
            
        return pg_loss.item(), v_loss.item(), entropy_loss.item()

    def train(self):
        """The Main Training Loop"""
        print(f"[PPO] Starting training for {self.algo_cfg.total_timesteps} steps...")
        obs, _ = self.env.reset()
        
        while self.global_step < self.algo_cfg.total_timesteps:
            start_time = time.time()
            
            # 1. Collect Data
            next_obs = self.collect_rollout()
            
            # 2. Compute Advantages
            advantages, returns = self.compute_advantages(next_obs)
            
            # 3. Update Network
            pg_loss, v_loss, ent_loss = self.update(advantages, returns)
            
            # Logging
            sps = int((self.algo_cfg.num_steps * self.num_envs) / (time.time() - start_time))
            print(f"Step: {self.global_step} | SPS: {sps} | Value Loss: {v_loss:.4f} | Policy Loss: {pg_loss:.4f}")
            
            if self.wandb_run:
                self.wandb_run.log({
                    "charts/SPS": sps,
                    "losses/value_loss": v_loss,
                    "losses/policy_loss": pg_loss,
                    "losses/entropy": ent_loss
                }, step=self.global_step)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "global_step": self.global_step,
            "agent": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        print(f"[PPO] Saved -> {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.global_step = ckpt["global_step"]
        self.agent.load_state_dict(ckpt["agent"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"[PPO] Loaded <- {path}")