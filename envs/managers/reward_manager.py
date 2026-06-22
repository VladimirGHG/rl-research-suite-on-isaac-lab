"""
Reward manager configuration.

FIX 1: API updated — 'isaac.lab' → 'isaaclab' (two separate bugs: import and func paths)
FIX 2: Non-existent MDP functions replaced:
        'isaac.lab.envs.mdp:object_ee_distance'     — does not exist
        'isaac.lab.envs.mdp:object_goal_coincidence' — does not exist
        Replaced with custom functions + real isaaclab MDP where available.

NOTE: Verify exact MDP function names on the GPU workstation by checking:
      isaaclab/envs/mdp/rewards.py
"""

import os
import torch
from isaaclab.managers import RewardTermCfg   # FIX: was 'isaac.lab'
from isaaclab.utils import configclass


# ── Custom reward term functions ──────────────────────────────────────────────
# These must return (num_envs,) tensors — NO Python for loops over envs
def reaching_reward(env, object_cfg_name: str = "object",
                    ee_cfg_name: str = "robot") -> torch.Tensor:
    """
    Distance-based reward: negative distance from end-effector to target.
    TODO Week 3: Replace placeholder with real asset position accessors.
    Verify exact accessor names from isaaclab docs on workstation.
    """
    # Placeholder — returns 0 reward until Week 3 asset accessors are confirmed
    return torch.zeros(env.num_envs, device=env.device)


def task_success(env, object_cfg_name: str = "object",
                 threshold: float = 0.02) -> torch.Tensor:
    """
    Binary success bonus: +1 when object within threshold of goal.
    TODO Week 3: Replace placeholder with real success condition.
    """
    return torch.zeros(env.num_envs, device=env.device)


# ── Reward manager config ─────────────────────────────────────────────────────
@configclass
class PlatformRewardsCfg:
    """
    Composable reward terms loaded from YAML.
    Adding a new term = one Python function + 3 lines of YAML.
    """

    reach_target: RewardTermCfg = RewardTermCfg(
        func=reaching_reward,
        weight=-1.0,
        params={"object_cfg_name": "object", "ee_cfg_name": "robot"},
    )

    # action_rate_l2 is a real isaaclab MDP function — verify on workstation
    action_penalty: RewardTermCfg = RewardTermCfg(
        func="isaaclab.envs.mdp:action_rate_l2",   # FIX: was 'isaac.lab.envs.mdp:...'
        weight=-0.01,
    )

    success_bonus: RewardTermCfg = RewardTermCfg(
        func=task_success,
        weight=10.0,
        params={"object_cfg_name": "object", "threshold": 0.02},
    )

    def __init__(self):
        super().__init__()
        self._apply_yaml_overrides()

    def _apply_yaml_overrides(self):
        """Load weight/func/params overrides from configs/reward/reward.yaml if present."""
        import yaml
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        yaml_path = os.path.join(repo_root, "configs", "reward", "reward.yaml")

        if not os.path.exists(yaml_path):
            return

        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            cfg = data.get("rewards", data) if isinstance(data, dict) else None
            if not cfg:
                return
            for term_name, specs in cfg.items():
                if hasattr(self, term_name):
                    term = getattr(self, term_name)
                    if "weight" in specs:
                        term.weight = specs["weight"]
                    if "params" in specs:
                        term.params.update(specs["params"])
            print(f"[RewardManager] Loaded overrides from {yaml_path}")
        except Exception as e:
            print(f"[RewardManager] Could not load reward.yaml: {e}. Using defaults.")
