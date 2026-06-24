import os
import yaml
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from envs.managers.common import get_ee_position


def reaching_reward(env, object_cfg_name: str = "object", ee_cfg_name: str = "robot") -> torch.Tensor:
    object_pos = env.scene[object_cfg_name].data.root_pos_w
    ee_pos = get_ee_position(env, ee_cfg_name)
    distance = torch.norm(object_pos - ee_pos, dim=-1)
    return torch.exp(-distance)


def task_success(env, object_cfg_name: str = "object", ee_cfg_name: str = "robot", threshold: float = 0.02) -> torch.Tensor:
    object_pos = env.scene[object_cfg_name].data.root_pos_w
    ee_pos = get_ee_position(env, ee_cfg_name)
    distance = torch.norm(object_pos - ee_pos, dim=-1)
    return (distance < threshold).float()


REWARD_FUNCTIONS = {
    "reaching_reward": reaching_reward,
    "task_success": task_success,
    "action_rate_l2": mdp.action_rate_l2,
}

_REWARD_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "reward", "reward.yaml",
)


@configclass
class PlatformRewardsCfg:
    def __post_init__(self):
        with open(_REWARD_YAML) as f:
            terms = yaml.safe_load(f)["rewards"]

        for term_name, spec in terms.items():
            func = REWARD_FUNCTIONS[spec["func"]]
            setattr(self, term_name, RewardTermCfg(
                func=func,
                weight=spec["weight"],
                params=spec.get("params", {}),
            ))
