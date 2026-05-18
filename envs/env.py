import gymnasium as gym
import torch
import numpy as np

try: # Attempt to get the actual ManagerBasedRLEnv, if available.
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
except ImportError:
    print("MOCK MODE")

    # Imitate Isaac Lab configuration.
    class ManagerBasedRLEnvCfg:
        def __init__(self):
            self.task_name = "mock_test"
            self.num_envs = 4
            self.viewer = [2.5, 2.5, 2.5]

    class ManagerBasedRLEnv:
        def __init__(self, cfg: ManagerBasedRLEnvCfg):
            self.cfg = cfg
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Imitate mock observation space.
            class MockObsManager:
                def compute(self):
                    return {"policy": torch.zeros((cfg.num_envs, 522))}
            
            # Imitate mock action space.
            class MockActionManager:
                def __init__(self):
                    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,))
            
            self.observation_manager = MockObsManager()
            self.action_manager = MockActionManager()

        def step(self, action):
            mock_obs_dict = self.observation_manager.compute()
            return mock_obs_dict, torch.zeros(self.cfg.num_envs), torch.zeros(self.cfg.num_envs), torch.zeros(self.cfg.num_envs), {}

        def reset(self):
            mock_obs_dict = self.observation_manager.compute()
            return mock_obs_dict, {}

        def close(self):
            pass

class IsaacLabPlatformEnv(ManagerBasedRLEnv):
    """
    Wrapper class on top of ManagerBasedRLEnv to connect any algorithm to Isaac Lab.
    """
    def __init__(self, cfg: ManagerBasedRLEnvCfg):
        super().__init__(cfg=cfg)
        
        with torch.no_grad(): # Inspect data without gradient tracking.
            initial_obs_dict = self.observation_manager.compute()
        
        raw_policy_tensor = initial_obs_dict["policy"] # Get the observation tensor for the policy from the observation manager's output.
        self.total_obs_dim = raw_policy_tensor.shape[-1] # Get the total observation dimension from the shape of the policy's observation tensor.

        # Define the observation and action spaces based on the initial observation and action manager, with dynamic shapes.
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32
        )
        self.action_space = self.action_manager.action_space

    def step(self, action: torch.Tensor): # Take an action in the environment, return the new observation, reward, done, and info.
        obs_dict, reward, terminated, truncated, extras = super().step(action)
        return obs_dict["policy"], reward, terminated, truncated, extras

    def reset(self): # Reset the envionment, return the initial observation, and start a new episode.
        obs_dict, extras = super().reset()
        return obs_dict["policy"], extras

if __name__ == "main":
    ex = IsaacLabPlatformEnv(ManagerBasedRLEnvCfg())
    print("Observation space:", ex.observation_space)