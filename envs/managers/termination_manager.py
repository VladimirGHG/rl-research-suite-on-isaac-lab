from omni.isaac.lab.managers import TerminationManagerCfg, TerminationTermCfg

class PlatformTerminationManagerCfg(TerminationManagerCfg):
    """Defines the logical boundaries that force an episode reset."""
    
    # Time limit termination term that ends the episode if a certain time limit is reached,
    # preventing infinite episodes and encouraging efficient task completion.
    timeout = TerminationTermCfg(
        func="omni.isaac.lab.envs.mdp:time_out", 
        time_out=True
    )
    
    # Safety violation termination term that ends the episode if the robot makes illegal contact with the environment,
    # such as colliding with the ground plane, promoting safe interactions within the environment.
    illegal_collision = TerminationTermCfg(
        func="omni.isaac.lab.envs.mdp:illegal_contact",
        params={"asset_cfg_name": "franka_panda", "target_contact_body": "ground_plane"}
    )
    
    # Success criteria termination term that ends the episode when the robot successfully completes the task,
    # such as bringing the target object within a certain distance of the goal location, providing a clear signal for task completion.
    task_completed = TerminationTermCfg(
        func="omni.isaac.lab.envs.mdp:success_criteria_met",
        params={"object_cfg_name": "target_cube", "threshold_distance": 0.02}
    )