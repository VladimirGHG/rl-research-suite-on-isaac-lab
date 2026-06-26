import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnostic script")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

def main():
    print("\n" + "="*60)
    print("STEP 1 — checking isaaclab_assets for Franka...")
    try:
        from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
        print("  OK: isaaclab_assets.robots.franka found")
    except ImportError:
        print("  MISS: trying thirdparty path...")
        import sys, os
        sys.path.insert(0, "thirdparty/Isaaclab/source/isaaclab_assets")
        try:
            from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
            print("  OK: found via thirdparty path")
        except ImportError:
            print("  NOT FOUND — list available robots:")
            import glob
            robots = glob.glob("thirdparty/**/robots/*.py", recursive=True)
            for r in robots:
                print(f"    {r}")

    print("\nSTEP 2 — checking mdp functions exist...")
    import isaaclab.envs.mdp as mdp
    for fn in ["joint_pos_rel", "joint_vel_rel", "time_out", "action_rate_l2", "is_alive"]:
        exists = hasattr(mdp, fn)
        print(f"  mdp.{fn}: {'OK' if exists else 'MISSING'}")

    print("\nSTEP 3 — checking TiledCamera import...")
    try:
        from isaaclab.sensors import TiledCameraCfg
        print("  OK: TiledCameraCfg found")
    except ImportError as e:
        print(f"  MISSING: {e}")

    print("\nSTEP 4 — building minimal env and printing Franka body names...")
    try:
        from envs.scene_cfg import FrankaManipulationSceneCfg
        from isaaclab.envs import ManagerBasedRLEnvCfg
        from isaaclab.managers import ObservationGroupCfg as ObsGroup
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.managers import RewardTermCfg as RewTerm
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        from isaaclab.utils import configclass
        import isaaclab.envs.mdp as mdp
        import isaaclab.sim as sim_utils

        @configclass
        class MinimalObsCfg:
            @configclass
            class PolicyCfg(ObsGroup):
                joint_pos = ObsTerm(func=mdp.joint_pos_rel)
                def __post_init__(self):
                    self.enable_corruption = False
                    self.concatenate_terms = True
            policy: PolicyCfg = PolicyCfg()

        @configclass
        class MinimalActionsCfg:
            from isaaclab.envs.mdp.actions import JointEffortActionCfg
            joint_effort = JointEffortActionCfg(asset_name="robot", joint_names=[".*"])

        @configclass
        class MinimalRewardsCfg:
            alive = RewTerm(func=mdp.is_alive, weight=1.0)

        @configclass
        class MinimalTermsCfg:
            time_out = DoneTerm(func=mdp.time_out, time_out=True)

        @configclass
        class DiagEnvCfg(ManagerBasedRLEnvCfg):
            """Diagnostic environment configuration for Isaac Lab."""

            scene: FrankaManipulationSceneCfg = FrankaManipulationSceneCfg(num_envs=1, env_spacing=2.5)
            observations: MinimalObsCfg = MinimalObsCfg()
            actions: MinimalActionsCfg = MinimalActionsCfg()
            rewards: MinimalRewardsCfg = MinimalRewardsCfg()
            terminations: MinimalTermsCfg = MinimalTermsCfg()
            def __post_init__(self):
                self.decimation = 2
                self.episode_length_s = 5.0
                self.sim.dt = 1/120

        env_cfg = DiagEnvCfg()
        env = ManagerBasedRLEnv(cfg=env_cfg)

        print("\n  === Franka body names ===")
        for name in env.scene["robot"].body_names:
            print(f"    {name}")

        print("\n  === Franka joint names ===")
        for name in env.scene["robot"].joint_names:
            print(f"    {name}")

        print("\n  === Action space ===")
        print(f"    {env.action_manager.action_term_dim}")

        print("\n  === Observation space ===")
        obs, _ = env.reset()
        print(f"    policy obs shape: {obs['policy'].shape}")

        env.close()
        print("\nDiagnostic PASSED — ready to train.")

    except Exception as e:
        import traceback
        print(f"\n  Diagnostic FAILED: {e}")
        traceback.print_exc()

    print("="*60)

if __name__ == "__main__":
    main()
    simulation_app.close()
