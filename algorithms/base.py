from abc import ABC, abstractmethod
from rl_base.policies.base import BasePolicy

class Trainer(ABC):
    """Abstract base class defining the uiversal minimal training algorithm interface."""

    def __init__(self, env, policy: BasePolicy, config):
        """
        Initialized the trainer.
        
        Args:
            env: An active instance of Isaac Lab environment
            policy: A policy determining our agent's action based on the observations.
            config: A Hydra-based nested configuration tree for the algorithm.
        """
        
        self.env = env
        self.policy = policy
        self.config = config

    @abstractmethod
    def collect_rollout(self) -> dict:
        """
        Collects new data from the environment to feed the agent.
        
        Returns:
            dict: Metrics obtained while collecting data.
        """
        raise NotImplementedError
    
    @abstractmethod
    def update(self) -> dict:
        """
        Updates the policy network parameters.
        
        Returns:
            dict: Training metrics from the update step.
        """
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self, num_of_episodes: int) -> dict:
        """Runs isolated validation checkpoints during the training.
        
        Args:
            num_of_episodes: Number of episodes to run evaluation on.
        
        Returns:
            dict: Metrics obtained from the evaluation.
        """
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path: str):
        """
        Serializes the current policy, algorithm's state and progress tracker
        into a unified Pytorch binary file.
        
        Args:
            path: The path for the save file.
        """
        raise NotImplementedError
    
    @abstractmethod
    def load(self, path: str):
        """Loads an algorithm with all it's properties.
        
        Args:
            path: The path to the saved binary file.
        """
        raise NotImplementedError
