import argparse
import os
import sys
from isaaclab.app import AppLauncher
import traceback
import gymnasium as gym
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# BOOT THE ENGINE FIRST
# This step dynamically injects 'pxr' and Omniverse paths into Python
parser = argparse.ArgumentParser(description="Custom Isaac Lab script.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import mdp

from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    ActionTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass

# Observations: a @configclass whose attributes are ObservationGroupCfg instances
@configclass
class MyObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # Each attribute here is an ObservationTermCfg
        joint_pos: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_pos)
        joint_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_vel)
    
    policy: PolicyCfg = PolicyCfg()

# Actions: a @configclass whose attributes are ActionTermCfg instances
@configclass
class MyActionsCfg:
    joint_effort: ActionTermCfg = ActionTermCfg(
        class_type=mdp.JointEffortAction,
        asset_name="terrain",
    )

# Rewards: a @configclass whose attributes are RewardTermCfg instances
@configclass
class MyRewardsCfg:
    alive: RewardTermCfg = RewardTermCfg(func=mdp.is_alive, weight=1.0)

# Terminations: a @configclass whose attributes are TerminationTermCfg instances
@configclass
class MyTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)


try: 
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    # All of these settings are going to be imported from the local .managers/ folder, 
    # where the wrappers for those setting are defined, to get their parameters from the .yaml files.
    # At the current stage this is just a test with the default manager classes to check whether the environment starts correctly.
    @configclass
    class MyEnvCfg(ManagerBasedRLEnvCfg):
        def __post_init__(self):
            self.decimation = 2
            self.episode_length_s = 10.0
            self.scene = InteractiveSceneCfg(num_envs=4, env_spacing=2.5, replicate_physics=True)
            self.observations = MyObservationsCfg()
            self.actions = MyActionsCfg()
            self.rewards = MyRewardsCfg()
            self.terminations = MyTerminationsCfg()
            super().__post_init__()  # call only once, at the end

except ImportError:
    print("MOCK MODE")
    traceback.print_exc()
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

if __name__ == "__main__":

    cfg = MyEnvCfg()    
    cfg.sim.device = "cuda:0"
    print("Initializing Isaac Lab environment...")
    ex = IsaacLabPlatformEnv(cfg=cfg)
    print("Observation space:", ex.observation_space)