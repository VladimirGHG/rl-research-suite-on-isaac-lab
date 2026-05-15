import torch
import gymnasium as gym
from abc import ABC, abstractmethod

class BasePolicy(ABC):
    """Abstract base class defining the universal minimal policy interface for creating and accessing policies."""

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box):
        """Policy's initilialization function should contain observation and action spaces as gymansium.spaces.Box
        which explicitly define mathematical constraints for the inputs and ouputs."""
        self.observation_space = observation_space # The inputs for the policy to expect
        self.action_space = action_space # The ouputs policy can generate

    @abstractmethod
    def predict(self, observations: torch.Tensor, deterministic: bool = False):
        """All the policies must be implemented SB3-style policy.predict(obs, deterministic) -> (action, state) API."""
        raise NotImplementedError
    
    def env_policy_actions_compatibility(self, env):
        """Checks the compatibility of environemnt's and policy's action and observation spaces.
        
        This is a crucial method to ensure that there will be no matrix dimension mismtach 
        between the policy's output and the environment's expected actions, 
        and environment's returned observations and the policy's expected observations. 
        
        It is checked using the .shape properties of both the policy's and active environment's action_space and observation_space."""
        if (env.observation_space.shape != self.observation_space.shape):
            raise ValueError("env and policy observation space mismatch" \
            f"Environment's provided observation shape: {env.observation_space.shape}" \
            f"Policy's expeceted observation space shape {self.observation_space.shape}")
        
        if (env.action_space.shape != self.action_space.shape):
            raise ValueError("env and policy action space mismatch" \
            f"Environment's expected action shape: {env.action_space.shape}" \
            f"Policy's action space shape {self.action_space.shape}")