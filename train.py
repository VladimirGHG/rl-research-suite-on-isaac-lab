import hydra
import torch
import gymnasium as gym
import wandb
from omegaconf import DictConfig, OmegaConf
from envs.env import IsaacLabPlatformEnv

try:
    from algorithms.ppo import PPOTrainer
    from algorithms.sac import SACTrainer
    from algorithms.td3 import TD3Trainer
except ImportError:
    # Local fallback definitions for laptop syntax checking
    class PPOTrainer:
        def __init__(self, env, algo_cfg, wandb_run): pass
        def train(self): print("MOCK TRAINING")
    SACTrainer = TD3Trainer = PPOTrainer

# Map Hydra configuration tags directly to your algorithm trainers
ALGO_REGISTRY = {
    "ppo": PPOTrainer,
    "sac": SACTrainer,
    "td3": TD3Trainer
}

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Init W&B with a dynamic run name based on the algorithm, environment, and seed for easy tracking in the dashboard.
    run_name = f"{cfg.algo.name}_{cfg.env.task_name}_seed{cfg.seed}"
    
    print(f"Connecting to live dashboard: {cfg.wandb.project}")
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    print(f"SETTING THE ENVIRONMENT: {cfg.env.task_name}")
    env = IsaacLabPlatformEnv(cfg=cfg.env)

    # Dynamically select the algorithm trainer class based on the configuration,
    # ensuring that only supported algorithms are instantiated.
    algo_key = str(cfg.algo.name).lower()
    if algo_key not in ALGO_REGISTRY:
        env.close()
        raise ValueError(f"Algorithm '{cfg.algo.name}' not supported. USE: {list(ALGO_REGISTRY.keys())}")

    print(f"Instantiating custom {cfg.algo.name} engine pipeline.")
    trainer = ALGO_REGISTRY[algo_key](env=env, algo_cfg=cfg.algo, wandb_run=run)

    try:
        trainer.train()
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise e
    finally:
        print("\nTERMINATING SESSION")
        env.close()
        wandb.finish()

if __name__ == "__main__":
    main()