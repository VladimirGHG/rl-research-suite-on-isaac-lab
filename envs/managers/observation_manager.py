from omni.isaac.lab.managers import ObservationManagerCfg, ObservationTermCfg
from encoders.resnet18 import get_platform_encoder
import torch

class PlatformObservationManagerCfg(ObservationManagerCfg):
    """Configuration for the observation manager, which defines how to compute the observations for the policy."""
    
    # Define the observation terms for the policy, from different components of the observation
    # that will be concatenated together to form the final observation vector for the policy.
    policy = ObservationTermCfg(
        # The policy observation term is a concatenation of joint positions, joint velocities, and resnet's returned features.
        joint_positions=ObservationTermCfg(
            func="omni.isaac.lab.envs.mdp:joint_pos",
            scale=1.0
        ),
        joint_velocities=ObservationTermCfg(
            func="omni.isaac.lab.envs.mdp:joint_vel",
            scale=0.1
        ),
        resnet_visual_features=ObservationTermCfg(
            func=get_platform_encoder(device="cuda" if torch.cuda.is_available() else "cpu"),
            params={"camera_cfg_name": "franka_wrist_camera"}
        )
    )