"""
Termination manager configuration.

VALIDATED against Isaac Lab official docs (main, June 2026):
- Plain @configclass — matches official cartpole TerminationsCfg pattern
- Functions as callables (func=mdp.time_out), not strings
- time_out=True marks episode limit as truncation (not termination)
  so SAC/PPO bootstrap value correctly for truncated episodes
"""

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass


# ── Custom termination term ────────────────────────────────────────────────────
def task_completed(env, object_cfg_name: str = "object",
                   threshold: float = 0.02) -> torch.Tensor:
    """
    End episode early when task is achieved.
    Returns (num_envs,) bool tensor — fully vectorised.
    TODO: replace placeholder with real success check in Week 3.
    """
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


# ── Termination config ─────────────────────────────────────────────────────────
@configclass
class PlatformTerminationsCfg:
    """Episode reset conditions."""

    # Episode time limit — truncation, not termination
    timeout: TerminationTermCfg = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )

    # Early success termination
    task_success: TerminationTermCfg = TerminationTermCfg(
        func=task_completed,
        params={"object_cfg_name": "object", "threshold": 0.02},
    )
