import torch


def get_ee_position(env, ee_cfg_name: str = "robot") -> torch.Tensor:
    """
    World-frame position of the Franka end-effector (panda_hand link), shape (num_envs, 3).
    Shared by reward and termination functions so "what counts as the end-effector"
    is defined in exactly one place.
    """
    robot = env.scene[ee_cfg_name]
    ee_body_idx, _ = robot.find_bodies("panda_hand")
    if len(ee_body_idx) == 0:
        raise ValueError(
            f"No body named 'panda_hand' found on asset '{ee_cfg_name}'. "
            f"Available body names: {robot.body_names}. "
            f"Update the body name in envs/managers/common.py to match."
        )
    return robot.data.body_pos_w[:, ee_body_idx[0]]
