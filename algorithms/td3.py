import copy
import os

import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import Trainer
from policies.base import BasePolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The Actor and Critic classes define the neural network architectures for the policy and value function approximators. 
# The Actor network takes in features extracted from observations and outputs actions, 
# while the Critic network evaluates the quality of state-action pairs. 
# The TD3Policy class encapsulates both networks and provides methods for action selection and prediction. 
class Actor(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(feature_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.l1(features))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(feature_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.l4 = nn.Linear(feature_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, features: torch.Tensor, action: torch.Tensor):
        fa = torch.cat([features, action], dim=1)
        q1 = F.relu(self.l1(fa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(fa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        fa = torch.cat([features, action], dim=1)
        q1 = F.relu(self.l1(fa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)


class TD3Policy(BasePolicy):
    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box, encoder, max_action: float, feature_dim: int = 512):
        super().__init__(observation_space, action_space)
        self.encoder = encoder
        self.max_action = max_action
        self.actor = Actor(feature_dim, action_space.shape[0], max_action).to(device)
        self.critic = Critic(feature_dim, action_space.shape[0]).to(device)

    # The get_action method processes pixel observations through the encoder to extract features, 
    # then passes those features through the actor network to produce an action. 
    # It ensures that the input pixel state has the correct batch dimension and operates in a no-gradient context 
    # since this is used for action selection during interaction with the environment.
    def get_action(self, pixel_state: torch.Tensor) -> torch.Tensor:
        """Processes pixel observations through the encoder to extract features, then passes those features through the actor network to produce an action."""
        if len(pixel_state.shape) == 3:
            pixel_state = pixel_state.unsqueeze(0)
        with torch.no_grad():
            features = self.encoder(pixel_state)
            action = self.actor(features)
        return action

    def predict(self, observations: torch.Tensor, deterministic: bool = False):
        """The predict method is a standard interface for policies, which takes in observations and a deterministic flag."""
        action = self.get_action(observations)
        return action, None

# The PixelReplayBuffer class manages the storage and sampling of experiences for training.
class PixelReplayBuffer(object):
    def __init__(self, num_envs: int, max_size: int, action_dim: int, device):
        self.max_size = max_size
        self.num_envs = num_envs
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((max_size, num_envs, 3, 84, 84), dtype=torch.float32, device=device)
        self.actions = torch.zeros((max_size, num_envs, action_dim), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((max_size, num_envs, 3, 84, 84), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((max_size, num_envs, 1), dtype=torch.float32, device=device)
        self.not_dones = torch.zeros((max_size, num_envs, 1), dtype=torch.float32, device=device)

    # The add method stores a new experience in the buffer, updating the current pointer and size accordingly.
    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state.detach()
        self.actions[self.ptr] = action.detach()
        self.next_states[self.ptr] = next_state.detach()
        self.rewards[self.ptr] = reward.unsqueeze(-1).detach()
        self.not_dones[self.ptr] = (~done).unsqueeze(-1).float().detach()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # The sample method retrieves a batch of experiences from the buffer, randomly selecting indices for both time steps and environments, 
    # and returns the corresponding states, actions, next states, rewards, and not-done flags.
    def sample(self, batch_size: int):
        step_idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        env_idx = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        return (
            self.states[step_idx, env_idx],
            self.actions[step_idx, env_idx],
            self.next_states[step_idx, env_idx],
            self.rewards[step_idx, env_idx],
            self.not_dones[step_idx, env_idx]
        )

# The main policy class, based on processing pixel observations through an encoder and then using the actor network to select actions.
class Td3Trainer(Trainer):
    def __init__(self, env, policy: TD3Policy, config):
        super().__init__(env=env, policy=policy, config=config)
        
        self.policy: TD3Policy = policy
        
        self.discount = config.get("discount", 0.99)
        self.tau = config.get("tau", 0.005)
        self.policy_noise = config.get("policy_noise", 0.2)
        self.noise_clip = config.get("noise_clip", 0.5)
        self.policy_freq = config.get("policy_freq", 2)
        self.batch_size = config.get("batch_size", 256)
        self.expl_noise = config.get("expl_noise", 0.1)
        self.learning_starts = config.get("learning_starts", 5000)
        self.total_timesteps = config.get("total_timesteps", 1000000)
        self.checkpoint_interval = config.get("checkpoint_interval", 50000)
        
        self.action_dim = env.action_space.shape[0]
        self.num_envs = env.cfg.num_envs
        self.total_it = 0
        self.total_steps_collected = 0

        self.actor_target = copy.deepcopy(self.policy.actor)
        self.critic_target = copy.deepcopy(self.policy.critic)

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=config.get("lr_actor", 3e-4))
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=config.get("lr_critic", 3e-4))

        self.replay_buffer = PixelReplayBuffer(
            num_envs=self.num_envs,
            max_size=config.get("buffer_capacity", 50000),
            action_dim=self.action_dim,
            device=device
        )
        self.current_obs, _ = self.env.reset()

    def collect_rollout(self) -> dict:
        raw_state_tensor = self.current_obs["policy"].to(device)
        
        if self.total_steps_collected < self.learning_starts:
            actions = torch.rand((self.num_envs, self.action_dim), device=device) * 2.0 - 1.0
            actions = actions * self.policy.max_action
        else:
            actions = self.policy.get_action(raw_state_tensor)
            noise = torch.randn_like(actions) * self.expl_noise
            actions = (actions + noise).clamp(-self.policy.max_action, self.policy.max_action)

        next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = terminated | truncated

        self.replay_buffer.add(
            raw_state_tensor, 
            actions, 
            next_obs["policy"].to(device), 
            rewards.to(device), 
            dones.to(device)
        )
        
        self.current_obs = next_obs
        self.total_steps_collected += self.num_envs
        
        return {
            "mean_rollout_reward": rewards.mean().item(),
            "buffer_current_size": self.replay_buffer.size * self.num_envs,
            "total_steps_collected": self.total_steps_collected
        }

    def update(self) -> dict:
        if self.total_steps_collected < self.learning_starts or self.replay_buffer.size * self.num_envs < self.batch_size:
            return {"status": "Warm-up phase; skipping gradient updates."}

        self.total_it += 1
        metrics = {}
        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            state_features = self.policy.encoder(state)
            next_state_features = self.policy.encoder(next_state)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state_features) + noise).clamp(-self.policy.max_action, self.policy.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state_features, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.policy.critic(state_features, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        metrics["critic_loss"] = critic_loss.item()
        metrics["mean_q_estimation"] = current_Q1.mean().item()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.policy.critic.Q1(state_features, self.policy.actor(state_features)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            metrics["actor_loss"] = actor_loss.item()

            for param, target_param in zip(self.policy.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.policy.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return metrics

    def evaluate(self, num_of_episodes: int) -> dict:
        total_rewards = []
        for _ in range(num_of_episodes):
            obs, _ = self.env.reset()
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
            
            while not done.all():
                with torch.no_grad():
                    actions = self.policy.get_action(obs["policy"].to(device))
                next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
                episode_rewards += rewards.to(device) * (~done)
                done |= (terminated | truncated)
                obs = next_obs
            total_rewards.append(episode_rewards.cpu().numpy())
        return {"eval_mean_reward": float(np.mean(total_rewards))}

    def save(self, path: str):
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
        checkpoint = torch.load(path, map_location=device)
        self.policy.actor.load_state_dict(checkpoint["actor_state"])
        self.policy.critic.load_state_dict(checkpoint["critic_state"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        self.total_it = checkpoint["total_it"]
        self.total_steps_collected = checkpoint.get("total_steps_collected", 0)