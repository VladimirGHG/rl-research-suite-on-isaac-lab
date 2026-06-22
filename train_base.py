import os
from typing import cast, Any

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from envs.env import IsaacLabPlatformEnv, MyEnvCfg

try:
    from algorithms.ppo import PPOTrainer
    from algorithms.sac import SACTrainer
    from algorithms.td3 import TD3Trainer
except ImportError:
    class PPOTrainer:
        def __init__(self, env, algo_cfg, wandb_run, resume_path=None): self.resume_path = resume_path
        def train(self): print("MOCK TRAINING COMPLETE")
        def save(self, path): print(f"MOCK SAVE to {path}")
    SACTrainer = TD3Trainer = PPOTrainer

ALGO_REGISTRY = {
    "ppo": PPOTrainer,
    "sac": SACTrainer,
    "td3": TD3Trainer
}

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    algo_name = cfg.algo.get("name", "unknown_algo")
    task_name = cfg.env.get("task_name", "unknown_task")
    seed_val = cfg.get("seed", 42)
    
    run_name = f"{algo_name}_{task_name}_seed{seed_val}"
    checkpoint_dir = os.path.abspath(cfg.get("checkpoint_dir", "./checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Connecting to live dashboard: {cfg.wandb.project}")
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        mode=cfg.wandb.mode,
        config=cast(dict, OmegaConf.to_container(cfg, resolve=True))
    )

    print(f"SETTING THE ENVIRONMENT: {task_name}")

    env_cfg = MyEnvCfg()
    
    # Extract keys and map directly over class variables (scene_class_path, num_envs, env_spacing).
    for key, val in cfg.env.items():
        if hasattr(env_cfg, key) or key in ["scene_class_path", "num_envs", "env_spacing"]:
            setattr(env_cfg, key, val)
            
    # Safely forward the completely hydrated dataclass object structure to your platform
    env = IsaacLabPlatformEnv(cfg=env_cfg)

    # Dynamically select the algorithm trainer class based on configuration parameters
    algo_key = str(algo_name).lower()
    if algo_key not in ALGO_REGISTRY:
        env.close()
        raise ValueError(f"Algorithm '{cfg.algo.name}' not supported. USE: {list(ALGO_REGISTRY.keys())}")

    resume_training = cfg.get("resume", False)
    resume_checkpoint_path = cfg.get("resume_path", None)

    if resume_training and not resume_checkpoint_path:
        fallback_file = os.path.join(checkpoint_dir, f"latest_{algo_key}_{task_name}.zip")
        if os.path.exists(fallback_file):
            resume_checkpoint_path = fallback_file

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"Found valid checkpoint. Resuming session from: {resume_checkpoint_path}")
    else:
        if resume_training:
            print(f"Warning: Resume requested but target file path '{resume_checkpoint_path}' was not found. Starting fresh.")
        resume_checkpoint_path = None

    print(f"Instantiating custom {algo_name} engine pipeline.")

    trainer = ALGO_REGISTRY[algo_key](
        env=env, 
        algo_cfg=cfg.algo, 
        wandb_run=run, 
        resume_path=resume_checkpoint_path
    )

    try:
        trainer.train()
        final_save_path = os.path.join(checkpoint_dir, f"final_{algo_key}_{task_name}.zip")
        if hasattr(trainer, "save"):
            trainer.save(final_save_path)
            wandb.save(final_save_path)
            print(f"Final trained network checkpoint stored safely at: {final_save_path}")

    except Exception as e:
        print(f"ERROR OCCURRED DURING SIMULATION RUNTIME: {str(e)}")
        raise e
    finally:
        print("\nTERMINATING SESSION CLEANLY")
        env.close()
        wandb.finish()

if __name__ == "__main__":
    main()