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

    parser = argparse.ArgumentParser(description="Custom Isaac Lab script.")
    AppLauncher.add_app_launcher_args(parser)

    args_cli, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from isaaclab.utils import configclass
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.envs.mdp.actions import JointEffortActionCfg

    from envs.managers.observation_manager import PlatformObservationsCfg
    from envs.managers.reward_manager import PlatformRewardsCfg
    from envs.managers.termination_manager import PlatformTerminationsCfg
    from isaaclab.envs.mdp.actions import JointPositionActionCfg
    @configclass
    class MyActionsCfg:
        joint_position: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )
    def load_scene_cfg_from_string(class_path: str):
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
        num_envs: int = 128
        env_spacing: float = 2.5

        def __post_init__(self):
            self.decimation = 2
            self.episode_length_s = 15.0

            scene_target_path = getattr(self, "scene_class_path", "envs.scene_cfg:FrankaManipulationSceneCfg")
            ResolvedSceneClass = load_scene_cfg_from_string(scene_target_path)
            self.scene = ResolvedSceneClass(
                num_envs=self.num_envs, env_spacing=self.env_spacing, replicate_physics=True
            )

            self.observations = PlatformObservationsCfg()
            self.actions = MyActionsCfg()
            self.rewards = PlatformRewardsCfg()
            self.terminations = PlatformTerminationsCfg()

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
                    return {"policy": torch.zeros((cfg.num_envs, 539))}

            class MockActionManager:
                def __init__(self):
                    self.total_action_dim = 9
                    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(9,))

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
    def __init__(self, cfg: ManagerBasedRLEnvCfg):
        super().__init__(cfg=cfg)

        with torch.no_grad():
            initial_obs = self.observation_manager.compute()["policy"]
        self.total_obs_dim = initial_obs.shape[-1]

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_manager.total_action_dim,), dtype=np.float32
        )

    def _get_synthetic_pixels(self) -> torch.Tensor:
        raw = self.scene["franka_wrist_camera"].data.output["rgb"]
        return raw.permute(0, 3, 1, 2).float() / 255.0
