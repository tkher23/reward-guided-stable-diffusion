from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights
from torchvision.transforms import v2

from models.model_oracle import PriceOracleResNet18


class OracleInference:
    """Utility for frozen clean-cost oracle inference and gradients."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.model = PriceOracleResNet18(pretrained=False).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.image_size = int(checkpoint.get("image_size", 224))
        self.weights = ResNet18_Weights.DEFAULT
        self.normalize = self.weights.transforms()
        self.resize = v2.Resize((self.image_size, self.image_size), antialias=True)
        self.to_tensor = v2.ToImage()
        self.to_dtype = v2.ToDtype(torch.float32, scale=True)

    def _preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        x = self.to_dtype(self.to_tensor(image.convert("RGB")))
        x = self.resize(x)
        return self.normalize(x)

    def _preprocess_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # Expects [B, 3, H, W] in [0, 1].
        x = self.resize(image_tensor)
        return self.normalize(x)

    @torch.no_grad()
    def score_pil_images(self, images: list[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self._preprocess_pil(img) for img in images], dim=0).to(self.device)
        pred_log = self.model(batch)
        return torch.exp(pred_log)

    def predict_log_price_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Differentiable log-price prediction from normalized image tensor in [0,1]."""
        x = self._preprocess_tensor(image_tensor.to(self.device))
        return self.model(x)
