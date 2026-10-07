from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
from transformers import CLIPModel


@dataclass
class FreezeConfig:
    freeze_vision_embeddings: bool = True
    freeze_text_embeddings: bool = True
    freeze_vision_layers: int = 8
    freeze_text_layers: int = 8


class CLIPFinetune(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        self.model_name = model_name
        self.clip = CLIPModel.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        return_loss: bool = True,
    ):
        return self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=return_loss,
        )

    @torch.no_grad()
    def get_image_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.clip.get_image_features(pixel_values=pixel_values)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @torch.no_grad()
    def get_text_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        feats = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


def freeze_lower_layers(model: CLIPFinetune, cfg: FreezeConfig) -> None:
    """
    Freeze lower CLIP layers for both vision and text towers.
    """

    # Freeze everything first? No. We only freeze selected lower parts.
    # Vision embeddings
    if cfg.freeze_vision_embeddings:
        for p in model.clip.vision_model.embeddings.parameters():
            p.requires_grad = False

    # Text embeddings
    if cfg.freeze_text_embeddings:
        for p in model.clip.text_model.embeddings.parameters():
            p.requires_grad = False

    # Vision encoder lower layers
    vision_layers = model.clip.vision_model.encoder.layers
    num_v = min(cfg.freeze_vision_layers, len(vision_layers))
    for i in range(num_v):
        for p in vision_layers[i].parameters():
            p.requires_grad = False

    # Text encoder lower layers
    text_layers = model.clip.text_model.encoder.layers
    num_t = min(cfg.freeze_text_layers, len(text_layers))
    for i in range(num_t):
        for p in text_layers[i].parameters():
            p.requires_grad = False


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }