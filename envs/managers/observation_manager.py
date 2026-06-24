import torch
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass
from envs.managers.common import get_ee_position
# A private variable to hold the cached model.
_CACHED_ENCODER = None

def encode_visual_observation(env, camera_cfg_name: str = "franka_wrist_camera") -> torch.Tensor:
    global _CACHED_ENCODER

    from encoders.resnet18 import get_platform_encoder
    raw_rgb = env.scene[camera_cfg_name].data.output["rgb"]
    pixels = raw_rgb.permute(0, 3, 1, 2).float() / 255.0

    if _CACHED_ENCODER is None:
        from encoders.resnet18 import get_platform_encoder
        _CACHED_ENCODER = get_platform_encoder(device=str(pixels.device))
        _CACHED_ENCODER.eval()

    with torch.no_grad():
        features = _CACHED_ENCODER(pixels)

    return features

def object_position(env, object_cfg_name: str = "object") -> torch.Tensor:
    """Object position in world frame: (num_envs, 3)"""
    return env.scene[object_cfg_name].data.root_pos_w


def ee_position(env, ee_cfg_name: str = "robot") -> torch.Tensor:
    """End-effector position in world frame: (num_envs, 3)"""
    return get_ee_position(env, ee_cfg_name)


def relative_position(env, object_cfg_name: str = "object", ee_cfg_name: str = "robot") -> torch.Tensor:
    """Vector from EE to object: (num_envs, 3)"""
    obj_pos = env.scene[object_cfg_name].data.root_pos_w
    ee_pos = get_ee_position(env, ee_cfg_name)
    return obj_pos - ee_pos

@configclass
class PlatformObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos_rel: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel_rel: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_vel_rel)
        last_action: ObservationTermCfg = ObservationTermCfg(func=mdp.last_action)
        object_pos: ObservationTermCfg = ObservationTermCfg(func=object_position)
        ee_pos: ObservationTermCfg = ObservationTermCfg(func=ee_position)
        relative_pos: ObservationTermCfg = ObservationTermCfg(func=relative_position)
        # visual_features: ObservationTermCfg = ObservationTermCfg(
        #     func=encode_visual_observation,
        #     params={"camera_cfg_name": "franka_wrist_camera"},
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
