"""
Frozen ResNet18 encoder.

Spec requirements (§7 — Things you will get wrong):
  ✓ requires_grad=False on ALL parameters
  ✓ .eval() mode permanently (BatchNorm running stats must NOT drift)
  ✓ torch.no_grad() in forward pass
  ✓ ImageNet mean/std normalization applied internally
  ✓ get_platform_encoder() returns singleton — loads to GPU only once

Input format:
  TiledCamera produces (B, H, W, 3) uint8.
  env._get_synthetic_pixels() converts to (B, 3, H, W) float [0,1].
  This encoder receives (B, 3, H, W) float [0,1] and normalizes internally.

FIX from previous version:
  - Removed SB3 BaseFeaturesExtractor inheritance (was causing constructor crash)
  - Added ImageNet normalization (was missing — ResNet18 features were wrong)
  - Fixed get_platform_encoder() (was crashing because constructor needed observation_space arg)
  - SB3 wrapper kept separately as FrozenResNet18SB3 for the smoke test only
"""

import torch
import torch.nn as nn
from torchvision import models


class FrozenResNet18(nn.Module):
    """
    Standalone frozen ResNet18 feature extractor.
    Input:  (B, 3, H, W) float32 in [0, 1]
    Output: (B, 512) feature vectors

    Used by PPOTrainer, SACTrainer, TD3Trainer directly.
    NOT SB3-coupled.
    """

    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])

        # Freeze ALL parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # CRITICAL: eval mode prevents BatchNorm running stats from drifting
        self.feature_extractor.eval()

        # ImageNet normalization constants
        # ResNet18 was trained with these — applying them is non-optional
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float32 in [0, 1]
        Returns:
            (B, 512) feature vectors
        """
        with torch.no_grad():
            x = (x - self.mean) / self.std           # ImageNet normalization
            features = self.feature_extractor(x)     # (B, 512, 1, 1)
            return torch.flatten(features, start_dim=1)  # (B, 512)

    def train(self, mode: bool = True):
        """Override — encoder is ALWAYS in eval mode, never training mode."""
        return super().train(False)


# ── SB3 compatibility wrapper ─────────────────────────────────────────────────
# Only used for the SB3 PPO smoke test in env.py __main__
# Custom trainers (PPO/SAC/TD3) use FrozenResNet18 above directly
try:
    import gymnasium as gym
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class FrozenResNet18SB3(BaseFeaturesExtractor):
        """
        SB3-compatible wrapper. Only for the smoke test.
        Requires observation_space as first arg (SB3 convention).
        """
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
            super().__init__(observation_space, features_dim)
            self._encoder = FrozenResNet18()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self._encoder(x)

except ImportError:
    FrozenResNet18SB3 = None


# ── Global singleton ──────────────────────────────────────────────────────────
_GLOBAL_ENCODER = None


def get_platform_encoder(device: str) -> FrozenResNet18:
    """
    Returns the global singleton FrozenResNet18.
    Loads to device only once — reused on all subsequent calls.

    FIX: previous version crashed because FrozenResNet18 required observation_space
    (it inherited from SB3's BaseFeaturesExtractor). Now it is a plain nn.Module.
    """
    global _GLOBAL_ENCODER
    if _GLOBAL_ENCODER is None:
        _GLOBAL_ENCODER = FrozenResNet18().to(device)
    return _GLOBAL_ENCODER
