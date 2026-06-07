import os
import yaml

from isaac.lab.managers import RewardManagerCfg, RewardTermCfg
from isaaclab.utils import configclass

@configclass
class PlatformRewardManagerCfg(RewardManagerCfg):
    """Returns the total step reward. (CONFIGURABLE from rl_base/configs/reward.yaml)"""
    
    # Define the reward terms for the policy, which are different components of the reward 
    # that will be computed and summed together to form the final reward signal for the policy.

    # Distance-based reward term that encourages the robot to minimize the distance between the end-effector and the target object.
    reach_target = RewardTermCfg(
        func="isaac.lab.envs.mdp:object_ee_distance",
        weight=-1.0,
        params={"object_cfg_name": "target_cube", "ee_cfg_name": "franka_effector"}
    )
    # Action penalty term that discourages large or rapid changes in the robot's actions, promoting smoother and more efficient movements.
    action_penalty = RewardTermCfg(
        func="isaac.lab.envs.mdp:action_rate_l2",
        weight=-0.01
    )
    # Success bonus term that provides a positive reward when the robot successfully achieves the task.
    success_bonus = RewardTermCfg(
        func="isaac.lab.envs.mdp:object_goal_coincidence",
        weight=10.0
    )

    def __init__(self):
        """Reads reward.yaml at initialization and updates reward term configurations accordingly."""
        super().__init__()
        
        current_file_path = os.path.abspath(__file__)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        yaml_path = os.path.join(repo_root, "configs", "reward", "reward.yaml")
        
        # If file is missing, use defaults.
        if not os.path.exists(yaml_path):
            print(f"[RewardManager] 'reward.yaml' not found at {yaml_path}. Using internal defaults.")
            return

        try:
            with open(yaml_path, "r") as f:
                full_yaml = yaml.safe_load(f)
                
            # Grab either the top-level contents, or a sub-key named 'rewards' if it is wrapped.
            yaml_dict = full_yaml.get("rewards", full_yaml) if isinstance(full_yaml, dict) else None
            
            if not yaml_dict:
                print("[RewardManager] 'reward.yaml' is empty or invalid. Using internal defaults.")
                return

            # Iterate through the keys in the YAML and update the corresponding reward term configurations if they exist.
            for term_name, specs in yaml_dict.items():
                if hasattr(self, term_name):
                    term_cfg = getattr(self, term_name)
                    
                    if "weight" in specs:
                        term_cfg.weight = specs["weight"]
                    if "func" in specs:
                        term_cfg.func = specs["func"]
                    if "params" in specs and isinstance(specs["params"], dict):
                        term_cfg.params.update(specs["params"])
                        
            print(f"[RewardManager] Successfully loaded live configuration from {yaml_path}")
            
        except Exception as e:
            print(f"[RewardManager] Error loading 'reward.yaml': {e}. Using internal defaults.")
