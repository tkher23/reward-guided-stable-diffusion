from __future__ import annotations

import torch

from preference.models.preference_oracle import PreferenceOracle
from preference.models.feature_extractor import DifferentiableFeatureExtractor


class PreferenceOracleInference:
    """
    Wrapper combining PreferenceOracle + DifferentiableFeatureExtractor.
    Provides differentiable preference score for image tensors.
    """

    def __init__(
        self,
        preference_oracle: PreferenceOracle,
        feature_extractor: DifferentiableFeatureExtractor,
        device: str = "cuda",
    ) -> None:
        self.oracle = preference_oracle
        self.feature_extractor = feature_extractor
        self.device = device

    def predict_preference_score_tensor(
        self, image_tensor: torch.Tensor, user_preferences: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute differentiable preference score for image tensor.

        Args:
            image_tensor: [B, 3, H, W] or [3, H, W] in [0, 1]
            user_preferences: [B, num_features] or [num_features] user weights

        Returns:
            [B] or scalar preference score (differentiable)
        """
        # Extract features (differentiable)
        features = self.feature_extractor.extract_features_from_tensor(image_tensor)

        # Compute preference score (differentiable)
        score = self.oracle(features, user_preferences)

        return score

    def get_theta(self) -> torch.Tensor:
        """Get current shared theta."""
        return self.oracle.get_theta()

    def update_theta(self, gradient: torch.Tensor, learning_rate: float = 0.01):
        """Update shared theta."""
        self.oracle.update_theta(gradient, learning_rate=learning_rate)

    def update_user_preferences(
        self, user_pref: torch.Tensor, gradient: torch.Tensor, learning_rate: float = 0.05
    ) -> torch.Tensor:
        """
        Update user preference vector based on feedback.

        Args:
            user_pref: [num_features] current user preference
            gradient: [num_features] update signal (e.g., image features or -image features)
            learning_rate: step size

        Returns:
            updated user preference
        """
        updated = user_pref + learning_rate * gradient
        # Normalize to unit norm
        updated = updated / (updated.norm() + 1e-8)
        return updated
