from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class PreferenceOracle(nn.Module):
    """
    Differentiable preference model: scores images based on shared theta and user-specific u_A.

    preference_score = theta · features(image) + u_A · features(image)
                     = (theta + u_A) · features(image)
    """

    def __init__(self, num_features: int = 3, device: str = "cuda") -> None:
        super().__init__()
        self.num_features = num_features
        self.device = device

        # Shared preference model (learned from aggregate user interactions)
        self.theta = nn.Parameter(
            torch.ones(num_features, device=device) / num_features
        )

        self.to(device)

    def forward(
        self, features: torch.Tensor, user_preferences: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute preference score for features.

        Args:
            features: [B, num_features] feature vectors (e.g., CLIP similarities)
            user_preferences: [B, num_features] or [num_features] per-user weights
                              If None, uses only theta

        Returns:
            [B] or scalar preference scores
        """
        features = features.to(self.device)

        # Combined preference direction: shared theta plus the per-user vector,
        # i.e. score = (theta + u_A) . features.
        weight = self.theta
        if user_preferences is not None:
            weight = weight + user_preferences.to(self.device)

        # Dot product over the feature dimension. Robust to features shaped
        # [num_features] (-> scalar score) or [B, num_features] (-> [B]), and
        # differentiable w.r.t. features so gradients flow to the latent.
        score = (features * weight).sum(dim=-1)
        return score

    def update_theta(self, gradient: torch.Tensor, learning_rate: float = 0.01) -> None:
        """Gradient-ASCENT step on theta.

        The caller passes +features for a clicked image and -features for a
        skipped one, so theta moves *toward* the features of liked images
        (preference_score = theta . features is maximized). This matches the
        per-user u_A update, which also adds the (signed) features.
        """
        with torch.no_grad():
            self.theta.data += learning_rate * gradient.to(self.theta.device)

    def normalize_theta(self) -> None:
        """Normalize theta to unit L2 norm.

        Click feedback only determines theta's *direction*; without this the
        accumulated magnitude grows with the number of interactions and would
        swamp the per-user u_A term at steering time.
        """
        with torch.no_grad():
            norm = self.theta.data.norm()
            if norm > 0:
                self.theta.data /= norm

    def get_theta(self) -> torch.Tensor:
        """Get current theta (detached)."""
        return self.theta.detach().clone()

    def set_theta(self, new_theta: torch.Tensor) -> None:
        """Set theta to a new value."""
        with torch.no_grad():
            self.theta.data = new_theta.to(self.device)

    def save(self, path: str) -> None:
        """Persist the learned preference model (theta) as small, portable JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(
                {
                    "theta": self.theta.detach().cpu().tolist(),
                    "num_features": self.num_features,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "PreferenceOracle":
        """Load a preference model saved with `save`."""
        with open(path) as f:
            data = json.load(f)
        oracle = cls(num_features=int(data["num_features"]), device=device)
        oracle.set_theta(torch.tensor(data["theta"], dtype=torch.float32))
        return oracle
