import importlib
import os
import sys
import traceback
import argparse

import gymnasium as gym
import torch
import numpy as np

try:
    from isaaclab.app import AppLauncher

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

    from isaaclab.envs.mdp.actions import JointEffortActionCfg

    # Actions: a @configclass whose attributes are ActionTermCfg instances
    @configclass
    class MyActionsCfg:
        joint_effort: JointEffortActionCfg = JointEffortActionCfg(
            asset_name="robot",
            joint_names=[".*"],
        )

    # Rewards: a @configclass whose attributes are RewardTermCfg instances
    try:
        from envs.managers.reward_manager import PlatformRewardManagerCfg
    except ImportError:
        print("Failed to import PlatformRewardManagerCfg, using a mock reward configuration.")
        @configclass
        class PlatformRewardManagerCfg:
            alive: RewardTermCfg = RewardTermCfg(func=mdp.is_alive, weight=1.0)
            def apply_yaml_overrides(self): pass

    # Terminations: a @configclass whose attributes are TerminationTermCfg instances
    @configclass
    class MyTerminationsCfg:
        time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)

    from isaaclab.assets import ArticulationCfg
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import TiledCameraCfg
    from thirdparty.Isaaclab.source.isaaclab_assets.isaaclab_assets.robots.cartpole import CARTPOLE_CFG
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

    # @configclass
    # class MySceneCfg(InteractiveSceneCfg):
    #     robot: ArticulationCfg = CARTPOLE_CFG.replace(
    #         prim_path="{ENV_REGEX_NS}/Robot"
    #     )

    def load_scene_cfg_from_string(class_path: str):
        """Dynamically loads an InteractiveSceneCfg class from a string description."""
        try:
            module_path, class_name = class_path.split(":")
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except Exception as e:
            print(f"[EnvCfg] Failed to dynamically load scene path '{class_path}': {e}")
            # Default back to standard baseline if module parsing fails
            from envs.scene_cfg import FrankaManipulationSceneCfg
            return FrankaManipulationSceneCfg
        
    # All of these settings are going to be imported from the local .managers/ folder, 
    # where the wrappers for those setting are defined, to get their parameters from the .yaml files.
    # At the current stage this is just a test with the default manager classes to check whether the environment starts correctly.
    @configclass
    class MyEnvCfg(ManagerBasedRLEnvCfg):
        scene_class_path: str = "envs.scene_cfg:FrankaManipulationSceneCfg"
        
        def __post_init__(self):
            self.decimation = 2
            self.episode_length_s = 10.0

            scene_target_path = getattr(self, "scene_class_path", "envs.scene_cfg:FrankaManipulationSceneCfg")
            ResolvedSceneClass = load_scene_cfg_from_string(scene_target_path)
            self.scene = ResolvedSceneClass(num_envs=128, env_spacing=2.5, replicate_physics=True)
            # A blank scene config, since the actual scene will be defined in the .yaml file and loaded by the SceneManager. (FOR TESTING)
            # self.scene = InteractiveSceneCfg(num_envs=4, env_spacing=2.5, replicate_physics=True)
            self.observations = MyObservationsCfg()
            self.actions = MyActionsCfg()

            self.rewards = PlatformRewardManagerCfg()
            self.terminations = MyTerminationsCfg()

            self.scene.robot_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera",
                update_period=0.0,
                height=84,
                width=84,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
                # Position the camera 2.5 meters away and 1.5 meters up looking back down
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(-2.5, 0.0, 1.5),
                    rot=(0.9945, 0.0, 0.1045, 0.0), # Direct quaternion orientation tilt
                    convention="ros" # Using ROS convention for easier interpretability (x forward, z up)
                )
            )
            super().__post_init__()

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
        # self.observation_space = gym.spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32
        # )

        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=(3, 84, 84), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_manager.total_action_dim,),
            dtype=np.float32,
        )
        print(f"Gym Observation Space forcefully set to: {self.observation_space.shape}")

    def _get_synthetic_pixels(self) -> torch.Tensor:
        """Helper to cleanly pull and format raw synthetic GPU buffers."""
        raw_pixels = self.scene["robot_camera"].data.output["rgb"] # Shape: [B, 84, 84, 3]
        
        processed_pixels = raw_pixels.permute(0, 3, 1, 2).float() / 255.0
        return processed_pixels

    def step(self, action: torch.Tensor): # Take an action in the environment, return the new observation, reward, done, and info.
        obs_dict, reward, terminated, truncated, extras = super().step(action)
        return {"policy": self._get_synthetic_pixels()}, reward, terminated, truncated, extras

    def reset(self): # Reset the envionment, return the initial observation, and start a new episode.
        obs_dict, extras = super().reset()
        return {"policy": self._get_synthetic_pixels()}, extras