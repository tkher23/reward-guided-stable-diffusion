from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class OracleOutputTransform:
    """Stores normalization used for oracle targets (log-price z-score)."""

    mean: float
    std: float


class PriceOracleResNet18(nn.Module):
    """Differentiable clean-image price oracle.

    Input: RGB image tensor in [0, 1], shape [B, 3, H, W].
    Output: normalized log-price, shape [B].
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)
