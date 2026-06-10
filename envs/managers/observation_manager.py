"""
Observation manager configuration.

FIX 1: API updated from 'omni.isaac.lab' to 'isaaclab'
FIX 2: get_platform_encoder() removed from class definition — was crashing on import
        before AppLauncher started. Encoder now called lazily inside function body.
FIX 3: Corrected ObservationGroupCfg class structure — was incorrectly nesting
        ObservationTermCfg inside another ObservationTermCfg.
"""

import torch
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg   # FIX: was omni.isaac.lab
from isaaclab.utils import configclass                                   # FIX: was omni.isaac.lab


# ── Observation term functions ────────────────────────────────────────────────
def encode_visual_observation(env, camera_cfg_name: str = "robot_camera") -> torch.Tensor:
    """
    Observation term: pull TiledCamera RGB and encode via frozen ResNet18.

    Called per-step by Isaac Lab's ObservationManager.
    Returns (num_envs, 512) feature vectors.

    FIX: encoder loaded lazily here (inside function body), NOT at class definition time.
    Previous version called get_platform_encoder() at class creation which crashed
    before AppLauncher had started.
    """
    from encoders.resnet18 import get_platform_encoder  # lazy import — safe here

    raw_rgb = env.scene[camera_cfg_name].data.output["rgb"]  # (B, H, W, 3) uint8
    pixels  = raw_rgb.permute(0, 3, 1, 2).float() / 255.0    # (B, 3, H, W) [0,1]
    encoder = get_platform_encoder(device=str(pixels.device))
    return encoder(pixels)   # (B, 512)


# ── Observation manager config ────────────────────────────────────────────────
@configclass
class PlatformObservationManagerCfg:
    """
    Observation vector = joint_pos + joint_vel + visual_features (ResNet512)
    Total ≈ 600 dimensions (within spec's <600 target).
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):   # FIX: was a single ObservationTermCfg with nested terms
        joint_pos: ObservationTermCfg = ObservationTermCfg(
            func="isaaclab.envs.mdp:joint_pos",   # FIX: was omni.isaac.lab.envs.mdp
            scale=1.0,
        )
        joint_vel: ObservationTermCfg = ObservationTermCfg(
            func="isaaclab.envs.mdp:joint_vel",   # FIX: was omni.isaac.lab.envs.mdp
            scale=0.1,
        )
        # 512-d visual features from wrist camera
        visual_features: ObservationTermCfg = ObservationTermCfg(
            func=encode_visual_observation,
            params={"camera_cfg_name": "robot_camera"},
        )

    policy: PolicyCfg = PolicyCfg()
