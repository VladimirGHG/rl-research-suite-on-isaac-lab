from omni.isaac.lab.managers import RewardManagerCfg, RewardTermCfg

class PlatformRewardManagerCfg(RewardManagerCfg):
    """Returns the total step reward."""
    
    # Define the reward terms for the policy, which are different components of the reward 
    # that will be computed and summed together to form the final reward signal for the policy.

    # Distance-based reward term that encourages the robot to minimize the distance between the end-effector and the target object.
    reach_target = RewardTermCfg(
        func="omni.isaac.lab.envs.mdp:object_ee_distance",
        weight=-1.0,
        params={"object_cfg_name": "target_cube", "ee_cfg_name": "franka_effector"}
    )
    # Action penalty term that discourages large or rapid changes in the robot's actions, promoting smoother and more efficient movements.
    action_penalty = RewardTermCfg(
        func="omni.isaac.lab.envs.mdp:action_rate_l2",
        weight=-0.01
    )
    # Success bonus term that provides a positive reward when the robot successfully achieves the task.
    success_bonus = RewardTermCfg(
        func="omni.isaac.lab.envs.mdp:object_goal_coincidence",
        weight=10.0
    )