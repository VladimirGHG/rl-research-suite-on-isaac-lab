# policies/custom/ppo_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOActorNetwork(nn.Module):
    """
    Independent Gaussian policy parameterized by state-dependent mean
    and state-independent (learnable) log_std.
    """
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        # Simple MLP structure matching standard continuous control tasks
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        
        # State-independent learnable standard deviation (initialized to exp(0)=1.0)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_mean(x)

    def get_action(self, x: torch.Tensor, action=None):
        """
        Outputs: action sample, log_prob of that action, and distribution entropy.
        """
        mean = self(x)
        std = self.actor_logstd.expand_as(mean).exp()
        normal = torch.distributions.Normal(mean, std)

        if action is None:
            # Crucially for PPO: sample without reparameterization trick
            # (unless specifically using something like ReparamPPO)
            action = normal.sample()

        return action, normal.log_prob(action).sum(dim=1), normal.entropy().sum(dim=1)


class PPOCriticNetwork(nn.Module):
    """
    Standard value function model: V(s) -> scalar value.
    """
    def __init__(self, obs_dim: int):
        super().__init__()
        # Simple MLP structure
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_value(x)