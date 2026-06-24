import torch
import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from envs.managers.common import get_ee_position


def task_completed(env, object_cfg_name: str = "object",
                   threshold: float = 0.02) -> torch.Tensor:
    """
    End episode early when task is achieved.
    Same distance/threshold logic as reward_manager.task_success, so the episode
    ends in the exact step the success bonus fires — not before, not after.
    Returns (num_envs,) bool tensor — fully vectorised.
    """
    object_pos = env.scene[object_cfg_name].data.root_pos_w
    ee_pos = get_ee_position(env, ee_cfg_name="robot")
    distance = torch.norm(object_pos - ee_pos, dim=-1)
    return distance < threshold


@configclass
class PlatformTerminationsCfg:
    """Episode reset conditions."""

    timeout: TerminationTermCfg = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )

    task_success: TerminationTermCfg = TerminationTermCfg(
        func=task_completed,
        params={"object_cfg_name": "object", "threshold": 0.02},
    )
