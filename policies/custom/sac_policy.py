import torch
from policies.base import BasePolicy


class SACPolicy(BasePolicy):
    def __init__(self, actor, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.actor = actor

    def predict(self, observations: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            action, _, mean = self.actor.get_action(observations)
        return (mean if deterministic else action), None
