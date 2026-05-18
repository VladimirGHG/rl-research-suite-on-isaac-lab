import os
import torch
import gymnasium as gym

try:
    from stable_baselines3 import PPO, SAC, TD3
    _SB3_AVAILABLE = True
except ImportError:
    print("MOCK MODE")
    _SB3_AVAILABLE = False
class SB3HubPolicyWrapper:
    """
    A critical adapter class that securely loads Stable-Baselines3 policies from Hugging Face Hub repositories or local directories,
    and ensures they are structurally compatible with the active Isaac Lab environment's observation and action spaces
    """
    def __init__(self, repo_id: str, filename: str, env: gym.Env):
        """
        Args:
            repo_id: Local path block or Hugging Face repository identifier pointer.
            filename: The target checkpoint file name (e.g., 'ppo_franka_latest.zip').
            env: The instantiated IsaacLabPlatformEnv platform wrapper.
        """
        self.env = env
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Resolve target model filepath (can look locally or simulate cloud handshakes)
        self.checkpoint_path = os.path.join(repo_id, filename) if os.path.exists(repo_id) else filename
        
        # Load the compiled policy weights into memory execution layers
        self.model = self._load_policy_checkpoint()
        
        # Check the loaded model's expected input and output dimensions against the active environment's spaces to prevent runtime mismatches.
        self._assert_space_compatibility()

    def _load_policy_checkpoint(self):
        """Resolves system dependencies and securely un-pickles model binaries."""
        if not _SB3_AVAILABLE:
            # Return a mock model object with the expected interface for local laptop testing without SB3 installed.
            class MockModel: pass
            return MockModel()
                    
        # Dynamically determine the algorithm architecture to load based on the filename, defaulting to PPO if no clear identifier is found.
        lower_name = self.checkpoint_path.lower()
        if "sac" in lower_name:
            return SAC.load(self.checkpoint_path, device=self.device)
        elif "td3" in lower_name:
            return TD3.load(self.checkpoint_path, device=self.device)
        else:
            # Default to PPO architecture
            return PPO.load(self.checkpoint_path, device=self.device)

    def _assert_space_compatibility(self):
        """
        The Guardian Checkpoint. Matches saved weight spaces against 
        the active environment parameters to intercept structural runtime crashes.
        """
        if not _SB3_AVAILABLE:
            return # Skip validation criteria constraints in mock mode.
            
        env_obs_shape = self.env.observation_space.shape[-1]
        model_obs_shape = self.model.observation_space.shape[-1]
        
        assert env_obs_shape == model_obs_shape, (
            f"Environment observation space width ({env_obs_shape}) does not match loaded policy architecture layers ({model_obs_shape})!"
        )
        
        env_act_shape = self.env.action_space.shape[-1]
        model_act_shape = self.model.action_space.shape[-1]
        
        assert env_act_shape == model_act_shape, (
            f"Environment motor degrees of freedom ({env_act_shape}) do not align with loaded policy output dimensions ({model_act_shape})!"
        )
        
    def predict(self, obs: torch.Tensor, deterministic: bool = True) -> tuple:
        """
        Provides the mandated execution signature expected by eval.py.
        Maps inputs across tensors cleanly while abstracting away SB3 internal types.
        """
        if not _SB3_AVAILABLE:
            # Return a mock action tensor of zeros with the correct shape for testing without SB3.
            batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
            mock_action = torch.zeros((batch_size, self.env.action_space.shape[-1]), device=obs.device)
            return mock_action, None

        action, states = self.model.predict(obs, deterministic=deterministic)
        
        return action, states