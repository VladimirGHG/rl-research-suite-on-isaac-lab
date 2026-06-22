"""
Observation manager configuration.

VALIDATED against Isaac Lab official docs (main, June 2026):
- Plain @configclass with nested PolicyCfg(ObservationGroupCfg)
- Functions as callables (func=mdp.joint_pos_rel), not strings
- ObservationTermCfg has no 'scale' param — removed
- __post_init__ sets concatenate_terms=True per official cartpole example
- Encoder loaded lazily at runtime, NOT at class definition time
"""

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass


# ── Visual feature observation term ───────────────────────────────────────────
def encode_visual_observation(env, camera_cfg_name: str = "robot_camera") -> torch.Tensor:
    """
    Pull TiledCamera RGB and encode via frozen ResNet18 → (num_envs, 512).
    Lazy encoder load — safe to call after AppLauncher has started.
    """
    from encoders.resnet18 import get_platform_encoder
    raw_rgb = env.scene[camera_cfg_name].data.output["rgb"]   # (B, H, W, 3) uint8
    pixels  = raw_rgb.permute(0, 3, 1, 2).float() / 255.0    # (B, 3, H, W) float [0,1]
    encoder = get_platform_encoder(device=str(pixels.device))
    return encoder(pixels)                                     # (B, 512)


# ── Observation config ─────────────────────────────────────────────────────────
@configclass
class PlatformObservationsCfg:
    """
    Observation = joint_pos_rel + joint_vel_rel + ResNet512 visual features.
    Total ≈ 9 + 9 + 512 = 530 dims — within spec's <600 target.
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # Joint positions relative to default pose
        joint_pos_rel: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_pos_rel)
        # Joint velocities
        joint_vel_rel: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_vel_rel)
        # ResNet18 visual features from wrist camera
        visual_features: ObservationTermCfg = ObservationTermCfg(
            func=encode_visual_observation,
            params={"camera_cfg_name": "robot_camera"},
        )

        def __post_init__(self):
            self.enable_corruption  = False
            self.concatenate_terms  = True   # flat vector output per official docs

    policy: PolicyCfg = PolicyCfg()
