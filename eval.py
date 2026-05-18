#!/usr/bin/env python3
import hydra
import torch
import numpy as np
from omegaconf import DictConfig

from envs.env import IsaacLabPlatformEnv
from policies.from_hub import SB3HubPolicyWrapper

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    cfg.env.num_envs = 1 # Force single environment for evaluation to ensure deterministic reward tracking.
    
    print(f"Instantiating evaluation environment: {cfg.env.task_name}")
    env = IsaacLabPlatformEnv(cfg=cfg.env)

    # Load the policy from the specified Hugging Face Hub repository and file.
    policy = SB3HubPolicyWrapper(
        repo_id=cfg.experiment.eval_repo,
        filename=cfg.experiment.eval_file,
        env=env
    )

    # Reset the environment to get the initial observation and info dictionary, and initialize reward tracking variables.
    obs, info = env.reset()
    episode_reward_accumulator = 0.0
    all_completed_rewards = []

    print(f"Executing {cfg.experiment.eval_steps}")
    
    # We strip all active gradients to speed up execution frame processing.
    with torch.no_grad():
        for step in range(cfg.experiment.eval_steps):
            
            # Generate the action from the policy, in an SB3 style.
            action, _states = policy.predict(obs, deterministic=True)
            
            # Take a step in the environment using the generated action, and receive the new observation, and other information.
            obs, reward, terminated, truncated, extras = env.step(action)
            
            # Accumulate the reward for the current episode, converting it to a numpy scalar for easier tracking and logging.
            episode_reward_accumulator += reward.detach().cpu().numpy().mean()

            # Check if current trial environment lifecycle has reached a conclusion
            if terminated.any() or truncated.any():
                all_completed_rewards.append(episode_reward_accumulator)
                print(f"Trial Episode {len(all_completed_rewards)} Complete. Total Reward: {episode_reward_accumulator:.4f}")
                
                # Reset metrics and state layout trackers for next trial loop
                episode_reward_accumulator = 0.0
                obs, info = env.reset()

    if len(all_completed_rewards) > 0:
        print(f"Episode completed: {len(all_completed_rewards)}")
    else:
        print("Episode did not complete")

    # Clean up the resources.
    env.close()

if __name__ == "__main__":
    main()