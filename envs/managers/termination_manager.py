"""
Termination manager configuration.

FIX: API updated from 'omni.isaac.lab' to 'isaaclab'
NOTE: Verify 'isaaclab.envs.mdp:time_out' exists on workstation.
      Real termination functions: isaaclab/envs/mdp/terminations.py
"""

import torch
from isaaclab.managers import TerminationTermCfg   # FIX: was omni.isaac.lab
from isaaclab.utils import configclass


def task_completed(env, object_cfg_name: str = "object",
                   threshold: float = 0.02) -> torch.Tensor:
    """
    Terminate episode early when task succeeds.
    Returns (num_envs,) bool tensor. NO Python for loops.
    TODO Week 3: Replace with real asset accessors.
    """
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


@configclass
class PlatformTerminationManagerCfg:
    """Episode reset conditions."""

    # Time limit — time_out=True marks as truncation (not true termination)
    # so GAE bootstraps the value correctly instead of treating it as terminal
    timeout: TerminationTermCfg = TerminationTermCfg(
        func="isaaclab.envs.mdp:time_out",   # FIX: was omni.isaac.lab
        time_out=True,
    )

    task_success: TerminationTermCfg = TerminationTermCfg(
        func=task_completed,
        params={"object_cfg_name": "object", "threshold": 0.02},
    )
