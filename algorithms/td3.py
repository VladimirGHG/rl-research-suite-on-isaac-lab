from algorithms.base import Trainer

class TD3Trainer(Trainer):
    def __init__(self, env, algo_cfg, wandb_run=None, resume_path=None):
        super().__init__(env, algo_cfg, wandb_run, resume_path)
    def collect_rollout(self): raise NotImplementedError
    def update(self): raise NotImplementedError
    def evaluate(self, n): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    def load(self, path): raise NotImplementedError
    def train(self): raise NotImplementedError("TD3 not yet implemented")
