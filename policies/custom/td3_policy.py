import copy
import os

import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import Trainer
from policies.base import BasePolicy

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

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
