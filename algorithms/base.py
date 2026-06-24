from abc import ABC, abstractmethod
from policies.base import BasePolicy   # FIX: was 'from rl_base.policies.base import BasePolicy'


class Trainer(ABC):
    """Abstract base class defining the universal minimal training algorithm interface."""

    def __init__(self, env, algo_cfg, wandb_run=None, resume_path=None):
        """
        Args:
            env:         Active IsaacLabPlatformEnv instance.
            algo_cfg:    Hydra config node for the algorithm (cfg.algo).
            wandb_run:   Optional active W&B run for metric logging.
            resume_path: Optional path to checkpoint to resume from.
        """
        self.env         = env
        self.config      = algo_cfg
        self.wandb_run   = wandb_run
        self.resume_path = resume_path

    @abstractmethod
    def collect_rollout(self) -> dict:
        """Collect a batch of experience from the environment."""
        raise NotImplementedError

    @abstractmethod
    def update(self) -> dict:
        """Update policy/value network parameters from collected data."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, num_of_episodes: int) -> dict:
        """Run deterministic evaluation episodes."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        """Serialize policy weights and optimizer state to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        """Restore policy weights and optimizer state from disk."""
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """Full training loop: collect + update until total_timesteps."""
        raise NotImplementedError
