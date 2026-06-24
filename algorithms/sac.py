import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algorithms.base import Trainer
from policies.custom.sac_policy import SACPolicy


class _ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, buffer_size: int, device: str):
        self.max_size = buffer_size
        self.device = device
        self.obs = torch.zeros((buffer_size, obs_dim), device=device)
        self.next_obs = torch.zeros((buffer_size, obs_dim), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.rewards = torch.zeros((buffer_size, 1), device=device)
        self.dones = torch.zeros((buffer_size, 1), device=device)
        self.ptr = 0
        self.size = 0

    def add(self, obs, next_obs, action, reward, done):
        n = obs.shape[0]
        idxs = torch.arange(self.ptr, self.ptr + n) % self.max_size
        self.obs[idxs] = obs.detach()
        self.next_obs[idxs] = next_obs.detach()
        self.actions[idxs] = action.detach()
        self.rewards[idxs] = reward.detach().unsqueeze(-1)
        self.dones[idxs] = done.detach().float().unsqueeze(-1)
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (self.obs[idxs], self.next_obs[idxs], self.actions[idxs],
                self.rewards[idxs], self.dones[idxs])

    def __len__(self):
        return self.size


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SACActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_low, action_high):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", ((action_high - action_low) / 2.0).float())
        self.register_buffer("action_bias", ((action_high + action_low) / 2.0).float())

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        deterministic_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_mean


class SACQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SACTrainer(Trainer):
    """
    Off-policy SAC trainer for Isaac Lab.

    Training loop (per step):
        1. collect_rollout() — 1 step in all num_envs environments, store in buffer
        2. update()          — (after learning_starts) sample from buffer,
                               update Q-networks, actor, alpha, targets
        3. Repeat until total_timesteps
    """

    def __init__(self, env, algo_cfg, wandb_run=None, resume_path=None):
        super().__init__(env, algo_cfg, wandb_run, resume_path)
        self.device = env.device
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.num_envs = getattr(env, "num_envs", 1)

        action_low = torch.tensor(env.action_space.low, device=self.device)
        action_high = torch.tensor(env.action_space.high, device=self.device)

        self.actor = SACActorNetwork(self.obs_dim, self.action_dim, action_low, action_high).to(self.device)
        self.qf1 = SACQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.qf2 = SACQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.qf1_target = SACQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.qf2_target = SACQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=algo_cfg.q_lr
        )
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=algo_cfg.policy_lr)

        self.target_entropy = -self.action_dim
        if algo_cfg.autotune:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=algo_cfg.q_lr)
        else:
            self.alpha = algo_cfg.alpha

        self.rb = _ReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            buffer_size=int(algo_cfg.buffer_size),
            device=self.device,
        )

        self.global_step = 0
        self.current_obs = None
        self.start_time = None

        if resume_path and os.path.exists(resume_path):
            self.load(resume_path)
            print(f"[SAC] Resumed from: {resume_path} (step {self.global_step})")

    def collect_rollout(self) -> dict:
        cfg = self.config

        if self.current_obs is None:
            obs_dict, _ = self.env.reset()
            self.current_obs = obs_dict["policy"]

        self.global_step += self.num_envs

        if self.global_step < int(cfg.learning_starts):
            actions = torch.tensor(
                np.array([self.env.action_space.sample() for _ in range(self.num_envs)]),
                dtype=torch.float32, device=self.device
            )
        else:
            with torch.no_grad():
                actions, _, _ = self.actor.get_action(self.current_obs)

        next_obs_dict, rewards, terminated, truncated, _ = self.env.step(actions)
        next_obs = next_obs_dict["policy"]

        self.rb.add(
            obs=self.current_obs,
            next_obs=next_obs,
            action=actions,
            reward=rewards.float(),
            done=terminated.float(),
        )

        self.current_obs = next_obs

        prev_step = self.global_step - self.num_envs
        if self.start_time and prev_step // 1000 != self.global_step // 1000:
            sps = int(self.global_step / (time.time() - self.start_time))
            print(f"[SAC] step={self.global_step} | buffer={len(self.rb)} | SPS={sps}")
            print(f"[SAC] reward min={rewards.min().item():.5f} mean={rewards.mean().item():.5f} max={rewards.max().item():.5f}")

        return {"global_step": self.global_step}

    def update(self) -> dict:
        cfg = self.config
        obs, next_obs, actions, rewards, dones = self.rb.sample(int(cfg.batch_size))

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.get_action(next_obs)
            qf1_next = self.qf1_target(next_obs, next_actions)
            qf2_next = self.qf2_target(next_obs, next_actions)
            min_qf_next = torch.min(qf1_next, qf2_next) - self.alpha * next_log_pi
            next_q_value = rewards + (1.0 - dones) * cfg.gamma * min_qf_next

        qf1_values = self.qf1(obs, actions)
        qf2_values = self.qf2(obs, actions)
        qf1_loss = F.mse_loss(qf1_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        metrics = {
            "losses/qf1_values": qf1_values.mean().item(),
            "losses/qf2_values": qf2_values.mean().item(),
            "losses/qf1_loss": qf1_loss.item(),
            "losses/qf2_loss": qf2_loss.item(),
            "losses/qf_loss": (qf_loss / 2.0).item(),
            "losses/alpha": self.alpha,
        }

        if self.global_step % int(cfg.policy_frequency) == 0:
            for _ in range(int(cfg.policy_frequency)):
                pi, log_pi, _ = self.actor.get_action(obs)
                qf1_pi = self.qf1(obs, pi)
                qf2_pi = self.qf2(obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if cfg.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(obs)
                    alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()
                    metrics["losses/alpha_loss"] = alpha_loss.item()

            metrics["losses/actor_loss"] = actor_loss.item()

        if self.global_step % int(cfg.target_network_frequency) == 0:
            for p, t in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                t.data.copy_(cfg.tau * p.data + (1 - cfg.tau) * t.data)
            for p, t in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                t.data.copy_(cfg.tau * p.data + (1 - cfg.tau) * t.data)

        return metrics

    def train(self):
        cfg = self.config
        total_timesteps = int(cfg.total_timesteps)
        ckpt_every = int(cfg.checkpoint_interval)
        eval_every = int(getattr(cfg, "eval_interval", 50000))
        eval_episodes = int(getattr(cfg, "eval_episodes", 5))
        self.start_time = time.time()

        print(f"[SAC] Starting | total_steps={total_timesteps} | "
              f"num_envs={self.num_envs} | learning_starts={cfg.learning_starts}")

        while self.global_step < total_timesteps:
            prev_step = self.global_step
            self.collect_rollout()

            if self.global_step > int(cfg.learning_starts):
                metrics = self.update()

                if self.wandb_run is not None and prev_step // 100 != self.global_step // 100:
                    metrics["charts/global_step"] = self.global_step
                    metrics["charts/SPS"] = int(self.global_step / (time.time() - self.start_time))
                    self.wandb_run.log(metrics, step=self.global_step)

            if prev_step // eval_every != self.global_step // eval_every:
                eval_metrics = self.evaluate(eval_episodes)
                print(f"[SAC] eval @ step={self.global_step} | "
                      f"mean_return={eval_metrics['eval/mean_return']:.3f} | "
                      f"std_return={eval_metrics['eval/std_return']:.3f}")
                if self.wandb_run is not None:
                    self.wandb_run.log(eval_metrics, step=self.global_step)
                self.current_obs = None

            if prev_step // ckpt_every != self.global_step // ckpt_every:
                self.save(f"checkpoints/sac_step_{self.global_step}.pt")

        print(f"[SAC] Training complete — {self.global_step} steps.")

    def evaluate(self, num_of_episodes: int) -> dict:
        returns = []
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"]
        ep_return = torch.zeros(self.num_envs, device=self.device)

        while len(returns) < num_of_episodes:
            with torch.no_grad():
                _, _, deterministic_action = self.actor.get_action(obs)
            obs_dict, reward, terminated, truncated, _ = self.env.step(deterministic_action)
            obs = obs_dict["policy"]
            ep_return += reward.float()
            done = terminated | truncated
            for i, d in enumerate(done):
                if d:
                    returns.append(ep_return[i].item())
                    ep_return[i] = 0.0

        return {
            "eval/mean_return": float(np.mean(returns[:num_of_episodes])),
            "eval/std_return": float(np.std(returns[:num_of_episodes])),
        }

    def to_policy(self) -> SACPolicy:
        return SACPolicy(self.actor, self.env.observation_space, self.env.action_space)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "global_step": self.global_step,
            "actor": self.actor.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "qf1_target": self.qf1_target.state_dict(),
            "qf2_target": self.qf2_target.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "alpha": self.alpha,
        }
        if self.config.autotune:
            ckpt["log_alpha"] = self.log_alpha.data
            ckpt["a_optimizer"] = self.a_optimizer.state_dict()
        torch.save(ckpt, path)
        print(f"[SAC] Saved -> {path} (step {self.global_step})")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.global_step = ckpt["global_step"]
        self.actor.load_state_dict(ckpt["actor"])
        self.qf1.load_state_dict(ckpt["qf1"])
        self.qf2.load_state_dict(ckpt["qf2"])
        self.qf1_target.load_state_dict(ckpt["qf1_target"])
        self.qf2_target.load_state_dict(ckpt["qf2_target"])
        self.q_optimizer.load_state_dict(ckpt["q_optimizer"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.alpha = ckpt["alpha"]
        if self.config.autotune and "log_alpha" in ckpt:
            self.log_alpha.data = ckpt["log_alpha"]
            self.a_optimizer.load_state_dict(ckpt["a_optimizer"])
        print(f"[SAC] Loaded from {path} (step {self.global_step})")
