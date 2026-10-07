from __future__ import annotations

import numpy as np
import torch


class SyntheticUser:
    """A synthetic user with known preferences that make click decisions."""

    def __init__(
        self,
        user_id: int,
        true_preference: torch.Tensor,
        click_offset: float = 0.0,
        click_gain: float = 8.0,
    ) -> None:
        """
        Args:
            user_id: unique identifier
            true_preference: [num_features] ground truth preference vector
            click_offset: alignment score at which click probability is 50%
                (set per user to the median alignment over the catalog so clicks
                are selective rather than near-universal)
            click_gain: steepness of the sigmoid click curve
        """
        self.user_id = user_id
        self.true_preference = (
            true_preference.clone().detach()
            if isinstance(true_preference, torch.Tensor)
            else torch.tensor(true_preference, dtype=torch.float32)
        )
        self.click_offset = click_offset
        self.click_gain = click_gain

        # Normalize preference to unit norm
        self.true_preference = (
            self.true_preference / self.true_preference.norm()
        )

        self.interaction_history = []

    def score_image(self, features: torch.Tensor) -> float:
        """
        Alignment of an image with the user's true preference.

        Returns:
            raw cosine similarity in [-1, 1]
        """
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu()
        else:
            features = torch.tensor(features, dtype=torch.float32)

        score = torch.nn.functional.cosine_similarity(
            features.unsqueeze(0), self.true_preference.unsqueeze(0)
        )
        return float(score.item())

    def decide_click(self, features: torch.Tensor, add_noise: bool = True) -> bool:
        """
        Decide whether to click on an image based on feature alignment.

        Click probability follows a logistic curve in the alignment score:
            P(click) = sigmoid(click_gain * (score - click_offset))
        With click_offset at the user's median catalog alignment this yields a
        graded, selective click model (~50% on typical images, higher on
        well-aligned ones), so steering toward the user's preference has room to
        raise the click rate.

        Args:
            features: [num_features] feature vector
            add_noise: if True, sample the click stochastically; if False, take
                the deterministic MAP decision (used for clean bootstrap signal)

        Returns:
            bool: True if user clicks, False otherwise
        """
        score = self.score_image(features)
        p_click = 1.0 / (1.0 + np.exp(-self.click_gain * (score - self.click_offset)))

        if add_noise:
            clicked = bool(np.random.random() < p_click)
        else:
            clicked = bool(p_click > 0.5)

        self.interaction_history.append(
            {
                "features": features.clone().detach()
                if isinstance(features, torch.Tensor)
                else torch.tensor(features),
                "score": score,
                "clicked": clicked,
            }
        )

        return clicked

    def get_interaction_count(self) -> int:
        """Return number of interactions."""
        return len(self.interaction_history)

    def get_clicked_images(self) -> list[torch.Tensor]:
        """Return features of all clicked images."""
        return [h["features"] for h in self.interaction_history if h["clicked"]]

    def get_skipped_images(self) -> list[torch.Tensor]:
        """Return features of all skipped images."""
        return [h["features"] for h in self.interaction_history if not h["clicked"]]

    def __repr__(self) -> str:
        return f"SyntheticUser(id={self.user_id}, pref={self.true_preference.tolist()}, interactions={len(self.interaction_history)})"


# Fixed seed for the shared population preference. Deliberately independent of
# the per-call diversity seed so train and held-out users share the SAME
# population component -- that is exactly what the shared theta is meant to
# learn and transfer.
SHARED_PREFERENCE_SEED = 20240101


def create_diverse_synthetic_users(
    num_users: int = 50,
    num_features: int = 3,
    seed: int = 42,
    feature_matrix: torch.Tensor | list[torch.Tensor] | None = None,
    shared_weight: float = 0.5,
) -> list[SyntheticUser]:
    """Create synthetic users as a shared population preference + individual taste.

        pref_i = normalize(shared_weight * shared + (1 - shared_weight) * individual_i)

    The `shared` component is common to all users (and identical across calls,
    via a fixed seed), so the shared theta learned from training users is a
    useful prior for held-out users. The `individual_i` component is what each
    user's u_A must discover. shared_weight in [0, 1] trades off how much of
    preference is population-level vs personal.

    `feature_matrix` (the dataset features, [N, num_features]) is used only to
    calibrate each user's click offset to their median alignment over the
    catalog, so clicks are selective rather than near-universal.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if feature_matrix is not None:
        if isinstance(feature_matrix, list):
            feature_matrix = torch.stack(feature_matrix)
        feature_matrix = feature_matrix.detach().cpu().float()
        num_features = feature_matrix.shape[1]
        catalog_unit = feature_matrix / feature_matrix.norm(
            dim=1, keepdim=True
        ).clamp_min(1e-8)

    # Shared population preference (same for every user across calls).
    shared_rng = np.random.default_rng(SHARED_PREFERENCE_SEED)
    shared = torch.tensor(
        shared_rng.normal(size=num_features), dtype=torch.float32
    )
    shared = shared / shared.norm()

    users = []
    for i in range(num_users):
        individual = torch.randn(num_features)
        individual = individual / individual.norm()
        pref = shared_weight * shared + (1.0 - shared_weight) * individual
        pref = pref / pref.norm()

        # Click offset = this user's median alignment over the catalog, so they
        # click ~half of typical images (selective). Falls back to 0.
        if feature_matrix is not None:
            click_offset = float((catalog_unit @ pref).median())
        else:
            click_offset = 0.0

        users.append(
            SyntheticUser(
                user_id=i,
                true_preference=pref,
                click_offset=click_offset,
            )
        )

    return users
