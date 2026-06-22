"""
SAC Trainer — adapted from CleanRL's sac_continuous_action.py for Isaac Lab.

Owner: Student B

Key differences from CleanRL:
  - Isaac Lab returns GPU tensors — no numpy/.cpu() conversion needed
  - Observations are (B, 3, 84, 84) pixels — encoded to 512-d via FrozenResNet18
  - Replay buffer stores 512-d encoded features (3000x smaller than raw pixels)
  - No gym.vector.SyncVectorEnv — Isaac Lab handles num_envs internally
  - Config from Hydra DictConfig (algo_cfg), not argparse/tyro
  - Logging to W&B, not TensorBoard
  - Follows platform's Trainer interface (collect_rollout / update / train)

Architecture (same as CleanRL SAC):
  - Actor:         squashed Gaussian with tanh, reparameterization trick
  - QNetwork x2:  twin critics to reduce Q-value overestimation
  - Target nets:  soft Polyak update every target_network_frequency steps
  - Entropy:      automatic alpha tuning (autotune=True by default)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models


# =============================================================================
# Standalone frozen ResNet18 encoder
# We define this inline because encoders/resnet18.py inherits from SB3's
# BaseFeaturesExtractor (requires observation_space arg) and get_platform_encoder
# is broken. This version has zero external dependencies beyond torchvision.
# =============================================================================
class _FrozenEncoder(nn.Module):
    """
    Standalone frozen ResNet18.
    Input:  (B, 3, H, W) float32 in [0, 1]
    Output: (B, 512) feature vectors

    Three critical rules from spec section 7:
      1. requires_grad=False on ALL params
      2. .eval() mode always — BatchNorm must never update running stats
      3. torch.no_grad() in forward pass
    """
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.net = nn.Sequential(*list(base.children())[:-1])

        # Rule 1: freeze all parameters
        for p in self.net.parameters():
            p.requires_grad = False

        # Rule 2: permanently in eval mode
        self.net.eval()

        # ImageNet normalization constants — ResNet18 was trained with these
        # Registered as buffers so they move with .to(device) automatically
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float32 in [0, 1]
               env._get_synthetic_pixels() handles uint8->float /255
        Returns:
            (B, 512) encoded feature vectors
        """
        with torch.no_grad():                            # Rule 3
            x = (x - self.mean) / self.std              # ImageNet normalisation
            features = self.net(x)                      # -> (B, 512, 1, 1)
            return torch.flatten(features, start_dim=1) # -> (B, 512)

    def train(self, mode: bool = True):
        """Always stay in eval mode — override train() to prevent mode changes."""
        return super().train(False)


# =============================================================================
# Simple GPU replay buffer
# Stores encoded 512-d features, NOT raw pixels.
# All tensors live on GPU — no CPU<->GPU transfers during sampling.
# =============================================================================
class _ReplayBuffer:
    """
    Circular replay buffer backed entirely by GPU tensors.

    Stores (obs, next_obs, action, reward, done) where obs is already
    encoded to 512-d by _FrozenEncoder.
    """
    def __init__(self, obs_dim: int, action_dim: int, buffer_size: int, device: str):
        self.max_size   = buffer_size
        self.device     = device

        # Pre-allocate everything on GPU once
        self.obs      = torch.zeros((buffer_size, obs_dim),    device=device)
        self.next_obs = torch.zeros((buffer_size, obs_dim),    device=device)
        self.actions  = torch.zeros((buffer_size, action_dim), device=device)
        self.rewards  = torch.zeros((buffer_size, 1),          device=device)
        self.dones    = torch.zeros((buffer_size, 1),          device=device)

        self.ptr  = 0
        self.size = 0

    def add(self, obs, next_obs, action, reward, done):
        """
        Add a batch of num_envs transitions.
        All inputs must already be GPU tensors.
        """
        n    = obs.shape[0]
        idxs = torch.arange(self.ptr, self.ptr + n) % self.max_size

        self.obs[idxs]      = obs.detach()
        self.next_obs[idxs] = next_obs.detach()
        self.actions[idxs]  = action.detach()
        self.rewards[idxs]  = reward.detach().unsqueeze(-1)
        self.dones[idxs]    = done.detach().float().unsqueeze(-1)

        self.ptr  = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size: int):
        """Sample batch_size random transitions. Returns all tensors on GPU."""
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[idxs],
            self.next_obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size


# =============================================================================
# SAC Actor — squashed Gaussian policy
# Architecture unchanged from CleanRL, input is 512-d encoded features
# =============================================================================
LOG_STD_MAX =  2
LOG_STD_MIN = -5


class SACActorNetwork(nn.Module):
    """
    Gaussian policy with tanh squashing.
    Input:  (B, 512)  — encoded obs features
    Output: action, log_prob, deterministic_mean
    """
    def __init__(self, obs_dim: int, action_dim: int,
                 action_low: torch.Tensor, action_high: torch.Tensor):
        super().__init__()
        self.fc1       = nn.Linear(obs_dim, 256)
        self.fc2       = nn.Linear(256, 256)
        self.fc_mean   = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

        # Action rescaling buffers
        # For Isaac Lab with action_space [-1, 1]: scale=1.0, bias=0.0
        # tanh already outputs [-1,1] so no real rescaling needed, but kept generic
        self.register_buffer("action_scale", ((action_high - action_low) / 2.0).float())
        self.register_buffer("action_bias",  ((action_high + action_low) / 2.0).float())

    def forward(self, x: torch.Tensor):
        x       = F.relu(self.fc1(x))
        x       = F.relu(self.fc2(x))
        mean    = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        # Clamp log_std to stable range (from SpinUp / Denis Yarats)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x: torch.Tensor):
        """
        Sample action + compute log probability using reparameterization trick.
        Returns: (action, log_prob, deterministic_mean)
        """
        mean, log_std = self(x)
        std    = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t    = normal.rsample()          # reparameterization: mu + std * N(0,1)
        y_t    = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Log prob with tanh squashing correction (SAC paper appendix)
        log_prob  = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob  = log_prob.sum(dim=1, keepdim=True)

        deterministic_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_mean


# =============================================================================
# SAC Q-Network — soft Q-function
# Two instances used (twin critics) to reduce Q-value overestimation
# =============================================================================
class SACQNetwork(nn.Module):
    """
    Q(s, a) -> scalar value.
    Input: concatenated (encoded_obs, action) = (512 + action_dim,)
    """
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# =============================================================================
# SAC Trainer
# =============================================================================
class SACTrainer:
    """
    Off-policy SAC trainer for Isaac Lab.

    Training loop (per step):
        1. collect_rollout() — 1 step in all num_envs environments,
                               encode obs, store in replay buffer
        2. update()          — (after learning_starts) sample from buffer,
                               update Q-networks, actor, alpha, targets
        3. Repeat until total_timesteps
    """

    def __init__(self, env, algo_cfg, wandb_run=None, resume_path=None):
        self.env       = env
        self.config    = algo_cfg
        self.wandb_run = wandb_run
        self.device    = env.device

        # Dimensions
        self.enc_dim    = 512
        self.action_dim = env.action_space.shape[0]
        self.num_envs   = getattr(env, "num_envs", 1)

        # Encoder
        self.encoder = _FrozenEncoder().to(self.device)

        # Networks
        action_low  = torch.tensor(env.action_space.low,  device=self.device)
        action_high = torch.tensor(env.action_space.high, device=self.device)

        self.actor      = SACActorNetwork(self.enc_dim, self.action_dim, action_low, action_high).to(self.device)
        self.qf1        = SACQNetwork(self.enc_dim, self.action_dim).to(self.device)
        self.qf2        = SACQNetwork(self.enc_dim, self.action_dim).to(self.device)
        self.qf1_target = SACQNetwork(self.enc_dim, self.action_dim).to(self.device)
        self.qf2_target = SACQNetwork(self.enc_dim, self.action_dim).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # Optimizers — only trainable params (encoder excluded automatically)
        self.q_optimizer     = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=algo_cfg.q_lr
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()),
            lr=algo_cfg.policy_lr
        )

        # Automatic entropy tuning
        self.target_entropy = -self.action_dim
        if algo_cfg.autotune:
            self.log_alpha   = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha       = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=algo_cfg.q_lr)
        else:
            self.alpha = algo_cfg.alpha

        # Replay buffer
        self.rb = _ReplayBuffer(
            obs_dim    = self.enc_dim,
            action_dim = self.action_dim,
            buffer_size= int(algo_cfg.buffer_size),
            device     = self.device,
        )

        # State
        self.global_step = 0
        self.current_obs = None
        self.start_time  = None

        # Resume
        if resume_path and os.path.exists(resume_path):
            self.load(resume_path)
            print(f"[SAC] Resumed from: {resume_path} (step {self.global_step})")

    def _encode(self, obs_dict) -> torch.Tensor:
        """Encode pixel dict from env to 512-d features."""
        pixels = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        return self.encoder(pixels.float())

    def collect_rollout(self) -> dict:
        """Take one step in all environments, store in buffer."""
        cfg = self.config

        if self.current_obs is None:
            obs_dict, _ = self.env.reset()
            self.current_obs = self._encode(obs_dict)

        self.global_step += self.num_envs

        # Action selection
        if self.global_step < int(cfg.learning_starts):
            actions = torch.tensor(
                np.array([self.env.action_space.sample() for _ in range(self.num_envs)]),
                dtype=torch.float32, device=self.device
            )
        else:
            with torch.no_grad():
                actions, _, _ = self.actor.get_action(self.current_obs)

        # Step environment
        next_obs_dict, rewards, terminated, truncated, _ = self.env.step(actions)
        next_encoded = self._encode(next_obs_dict)

        # Store terminated only (not truncated — we still bootstrap truncated value)
        self.rb.add(
            obs      = self.current_obs,
            next_obs = next_encoded,
            action   = actions,
            reward   = rewards.float(),
            done     = terminated.float(),
        )

        self.current_obs = next_encoded

        if self.start_time and self.global_step % 1000 == 0:
            sps = int(self.global_step / (time.time() - self.start_time))
            print(f"[SAC] step={self.global_step} | buffer={len(self.rb)} | SPS={sps}")

        return {"global_step": self.global_step}

    def update(self) -> dict:
        """Sample from buffer and update all networks."""
        cfg = self.config
        obs, next_obs, actions, rewards, dones = self.rb.sample(int(cfg.batch_size))

        # Soft Bellman target
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.get_action(next_obs)
            qf1_next     = self.qf1_target(next_obs, next_actions)
            qf2_next     = self.qf2_target(next_obs, next_actions)
            min_qf_next  = torch.min(qf1_next, qf2_next) - self.alpha * next_log_pi
            next_q_value = rewards + (1.0 - dones) * cfg.gamma * min_qf_next

        # Update Q-networks
        qf1_values = self.qf1(obs, actions)
        qf2_values = self.qf2(obs, actions)
        qf1_loss   = F.mse_loss(qf1_values, next_q_value)
        qf2_loss   = F.mse_loss(qf2_values, next_q_value)
        qf_loss    = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        metrics = {
            "losses/qf1_values": qf1_values.mean().item(),
            "losses/qf2_values": qf2_values.mean().item(),
            "losses/qf1_loss":   qf1_loss.item(),
            "losses/qf2_loss":   qf2_loss.item(),
            "losses/qf_loss":    (qf_loss / 2.0).item(),
            "losses/alpha":      self.alpha,
        }

        # Delayed actor + alpha update
        if self.global_step % int(cfg.policy_frequency) == 0:
            for _ in range(int(cfg.policy_frequency)):
                pi, log_pi, _ = self.actor.get_action(obs)
                qf1_pi     = self.qf1(obs, pi)
                qf2_pi     = self.qf2(obs, pi)
                min_qf_pi  = torch.min(qf1_pi, qf2_pi)
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

        # Polyak averaging for target networks
        if self.global_step % int(cfg.target_network_frequency) == 0:
            for p, t in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                t.data.copy_(cfg.tau * p.data + (1 - cfg.tau) * t.data)
            for p, t in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                t.data.copy_(cfg.tau * p.data + (1 - cfg.tau) * t.data)

        return metrics

    def train(self):
        """Full SAC training loop."""
        cfg             = self.config
        total_timesteps = int(cfg.total_timesteps)
        ckpt_every      = int(cfg.checkpoint_interval)
        self.start_time = time.time()

        print(f"[SAC] Starting | total_steps={total_timesteps} | "
              f"num_envs={self.num_envs} | learning_starts={cfg.learning_starts}")

        while self.global_step < total_timesteps:
            self.collect_rollout()

            if self.global_step > int(cfg.learning_starts):
                metrics = self.update()

                if self.global_step % 100 == 0 and self.wandb_run is not None:
                    metrics["charts/global_step"] = self.global_step
                    metrics["charts/SPS"] = int(
                        self.global_step / (time.time() - self.start_time)
                    )
                    self.wandb_run.log(metrics, step=self.global_step)

            if self.global_step % ckpt_every == 0:
                self.save(f"checkpoints/sac_step_{self.global_step}.pt")

        print(f"[SAC] Training complete — {self.global_step} steps.")

    def evaluate(self, num_of_episodes: int) -> dict:
        """Deterministic evaluation using mean action."""
        returns    = []
        obs_dict, _= self.env.reset()
        obs        = self._encode(obs_dict)
        ep_return  = torch.zeros(self.num_envs, device=self.device)

        while len(returns) < num_of_episodes:
            with torch.no_grad():
                _, _, deterministic_action = self.actor.get_action(obs)
            obs_dict, reward, terminated, truncated, _ = self.env.step(deterministic_action)
            obs        = self._encode(obs_dict)
            ep_return += reward.float()
            done = terminated | truncated
            for i, d in enumerate(done):
                if d:
                    returns.append(ep_return[i].item())
                    ep_return[i] = 0.0

        return {
            "eval/mean_return": float(np.mean(returns[:num_of_episodes])),
            "eval/std_return":  float(np.std(returns[:num_of_episodes])),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "global_step":     self.global_step,
            "actor":           self.actor.state_dict(),
            "qf1":             self.qf1.state_dict(),
            "qf2":             self.qf2.state_dict(),
            "qf1_target":      self.qf1_target.state_dict(),
            "qf2_target":      self.qf2_target.state_dict(),
            "q_optimizer":     self.q_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "alpha":           self.alpha,
        }
        if self.config.autotune:
            ckpt["log_alpha"]   = self.log_alpha.data
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
