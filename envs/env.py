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
    # AppLauncher must start before any other Isaac Lab imports.
    parser = argparse.ArgumentParser(description="Custom Isaac Lab script.")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of environments")
    AppLauncher.add_app_launcher_args(parser)

    # FIX 1: parse_known_args returns (namespace, remaining_list)
    # FIX 2: strip AppLauncher args from sys.argv so Hydra only sees key=value overrides
    args_cli, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from isaaclab.envs import mdp
    from isaaclab.utils import configclass
    from isaaclab.managers import (
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        TerminationTermCfg,
    )

    @configclass
    class MyObservationsCfg:
        @configclass
        class PolicyCfg(ObservationGroupCfg):
            joint_pos: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_pos)
            joint_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_vel)

        policy: PolicyCfg = PolicyCfg()

    from isaaclab.envs.mdp.actions import JointEffortActionCfg

    @configclass
    class MyActionsCfg:
        joint_effort: JointEffortActionCfg = JointEffortActionCfg(
            asset_name="robot",
            joint_names=[".*"],
        )

    # FIX 3: import PlatformRewardsCfg (new name from validated reward_manager.py)
    try:
        from envs.managers.reward_manager import PlatformRewardsCfg
    except ImportError:
        print("Failed to import PlatformRewardsCfg, using a mock reward configuration.")
        @configclass
        class PlatformRewardsCfg:
            alive: RewardTermCfg = RewardTermCfg(func=mdp.is_alive, weight=1.0)
            def apply_yaml_overrides(self): pass

    @configclass
    class MyTerminationsCfg:
        time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)

    from isaaclab.assets import ArticulationCfg
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

    # FIX 4: removed unused CARTPOLE_CFG import that crashed when thirdparty
    # submodule was not initialized. It was imported but never used (commented out).

    def load_scene_cfg_from_string(class_path: str):
        """Dynamically loads an InteractiveSceneCfg class from a string path."""
        try:
            module_path, class_name = class_path.split(":")
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except Exception as e:
            print(f"[EnvCfg] Failed to dynamically load scene path '{class_path}': {e}")
            from envs.scene_cfg import FrankaManipulationSceneCfg
            return FrankaManipulationSceneCfg

    @configclass
    class MyEnvCfg(ManagerBasedRLEnvCfg):
        scene_class_path: str = "envs.scene_cfg:FrankaManipulationSceneCfg"

        def __post_init__(self):
            self.decimation = 2
            self.episode_length_s = 10.0

            scene_target_path = getattr(self, "scene_class_path", "envs.scene_cfg:FrankaManipulationSceneCfg")
            ResolvedSceneClass = load_scene_cfg_from_string(scene_target_path)
            self.scene = ResolvedSceneClass(num_envs=128, env_spacing=2.5, replicate_physics=True)

            self.observations = MyObservationsCfg()
            self.actions = MyActionsCfg()
            self.rewards = PlatformRewardsCfg()
            self.terminations = MyTerminationsCfg()

            '''self.scene.robot_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand/rgb_camera",
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
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(-2.5, 0.0, 1.5),
                    rot=(0.9945, 0.0, 0.1045, 0.0),
                    convention="ros"
                )
            )'''
            super().__post_init__()

except ImportError:
    print("MOCK MODE")
    traceback.print_exc()

    class ManagerBasedRLEnvCfg:
        def __init__(self):
            self.task_name = "mock_test"
            self.num_envs = 4
            self.viewer = [2.5, 2.5, 2.5]

    class ManagerBasedRLEnv:
        def __init__(self, cfg: ManagerBasedRLEnvCfg):
            self.cfg = cfg
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            class MockObsManager:
                def compute(self):
                    return {"policy": torch.zeros((cfg.num_envs, 522))}

            class MockActionManager:
                def __init__(self):
                    self.total_action_dim = 7
                    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,))

            self.observation_manager = MockObsManager()
            self.action_manager = MockActionManager()

        def step(self, action):
            obs = self.observation_manager.compute()
            return obs, torch.zeros(self.cfg.num_envs), torch.zeros(self.cfg.num_envs), torch.zeros(self.cfg.num_envs), {}

        def reset(self):
            return self.observation_manager.compute(), {}

        def close(self):
            pass


class IsaacLabPlatformEnv(ManagerBasedRLEnv):
    """Wrapper over ManagerBasedRLEnv exposing Gymnasium-style spaces and pixel observations."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg):
        super().__init__(cfg=cfg)

        with torch.no_grad():
            initial_obs_dict = self.observation_manager.compute()

        raw_policy_tensor = initial_obs_dict["policy"]
        self.total_obs_dim = raw_policy_tensor.shape[-1]

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
        """Pull TiledCamera RGB and convert to (B, 3, H, W) float [0, 1]."""
        raw_pixels = self.scene["franka_wrist_camera"].data.output["rgb"]  # (B, 84, 84, 3)
        return raw_pixels.permute(0, 3, 1, 2).float() / 255.0

    def step(self, action: torch.Tensor):
        obs_dict, reward, terminated, truncated, extras = super().step(action)
        return {"policy": self._get_synthetic_pixels()}, reward, terminated, truncated, extras

    def reset(self):
        obs_dict, extras = super().reset()
        return {"policy": self._get_synthetic_pixels()}, extras
