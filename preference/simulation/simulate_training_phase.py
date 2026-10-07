from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from preference.models.preference_oracle import PreferenceOracle
from preference.simulation.synthetic_user_simulator import create_diverse_synthetic_users


def load_image_features(
    features_file: str = "sneakerdata/features/feature_labels.json",
    feature_space: str = "concepts",
    embeddings_file: str = None,
) -> tuple[list[str], list[torch.Tensor]]:
    """
    Load pre-extracted image features in the requested representation.

    Args:
        features_file: path to concept-score labels (feature_labels.json)
        feature_space: "concepts" -> per-attribute CLIP scores, z-scored across
            the dataset (low-dim, interpretable); "embedding" -> full
            L2-normalized CLIP image embedding (512-d).
        embeddings_file: path to image_embeddings.json (defaults to a sibling
            of features_file). Only used for feature_space="embedding".

    Returns:
        image_ids: list of image identifiers
        features_list: list of [num_features] tensors
    """
    if feature_space == "embedding":
        if embeddings_file is None:
            embeddings_file = str(
                Path(features_file).with_name("image_embeddings.json")
            )
        with open(embeddings_file) as f:
            embeddings = json.load(f)
        image_ids = list(embeddings.keys())
        # Embeddings are already L2-normalized at extraction time.
        features_list = [
            torch.tensor(embeddings[img_id], dtype=torch.float32)
            for img_id in image_ids
        ]
        return image_ids, features_list

    if feature_space != "concepts":
        raise ValueError(f"Unknown feature_space: {feature_space!r}")

    with open(features_file) as f:
        labels = json.load(f)

    image_ids = list(labels.keys())
    # concept_scores are CLIP zero-shot probabilities (see label_features_with_ai),
    # so they're already spread and comparable -- read them directly as f(x).
    features_list = [
        torch.tensor(labels[img_id]["concept_scores"], dtype=torch.float32)
        for img_id in image_ids
    ]
    return image_ids, features_list


def simulate_training_phase(
    num_synthetic_users: int = 50,
    interactions_per_user: int = 100,
    theta_learning_rate: float = 0.01,
    features_file: str = "sneakerdata/features/feature_labels.json",
    output_dir: str = "preference/results",
    seed: int = 42,
    feature_space: str = "concepts",
) -> tuple[PreferenceOracle, dict]:
    """
    Simulate training phase: 50 synthetic users interact with 100 images.
    Learn shared theta from aggregate click data.

    Returns:
        preference_oracle: trained PreferenceOracle with shared theta
        metadata: statistics from training
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load image features
    print(f"Loading image features (feature_space={feature_space})...")
    image_ids, features_list = load_image_features(
        features_file, feature_space=feature_space
    )
    print(f"Loaded {len(features_list)} images with {features_list[0].shape[0]}-d features")

    # Create synthetic users, each anchored to a real shoe in this feature space
    print(f"Creating {num_synthetic_users} synthetic users...")
    users = create_diverse_synthetic_users(
        num_users=num_synthetic_users,
        num_features=features_list[0].shape[0],
        seed=seed,
        feature_matrix=features_list,
    )

    # Initialize preference oracle
    num_features = features_list[0].shape[0]
    oracle = PreferenceOracle(num_features=num_features, device="cuda")

    # Training loop
    total_clicks = 0
    total_interactions = 0
    all_interactions = []

    print("\n--- TRAINING PHASE ---")
    for user_idx, user in enumerate(users):
        if user_idx % 10 == 0:
            print(f"User {user_idx}/{num_synthetic_users}")

        for _ in range(interactions_per_user):
            # Randomly select an image
            image_idx = np.random.randint(0, len(features_list))
            features = features_list[image_idx]

            # User decides whether to click
            clicked = user.decide_click(features, add_noise=True)

            # Record interaction
            interaction = {
                "user_id": user.user_id,
                "image_id": image_ids[image_idx],
                "features": features.tolist(),
                "clicked": clicked,
            }
            all_interactions.append(interaction)

            # Update shared theta from this interaction
            # If clicked: reinforce, else punish
            gradient = features.clone()
            if not clicked:
                gradient = -gradient

            oracle.update_theta(gradient, learning_rate=theta_learning_rate)

            if clicked:
                total_clicks += 1
            total_interactions += 1

    # Click feedback fixes theta's direction, not its scale; normalize so it
    # stays balanced against per-user u_A during steering.
    oracle.normalize_theta()

    print(f"\nTraining complete!")
    print(f"Total interactions: {total_interactions}")
    print(f"Total clicks: {total_clicks} ({100*total_clicks/total_interactions:.1f}%)")
    print(f"Final theta: {oracle.get_theta().tolist()}")

    # Save training metadata
    metadata = {
        "num_users": num_synthetic_users,
        "interactions_per_user": interactions_per_user,
        "total_interactions": total_interactions,
        "total_clicks": total_clicks,
        "click_rate": total_clicks / total_interactions,
        "final_theta": oracle.get_theta().tolist(),
        "user_preferences": [u.true_preference.tolist() for u in users],
    }

    metadata_file = output_path / "training_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_file}")

    return oracle, metadata


if __name__ == "__main__":
    oracle, metadata = simulate_training_phase()
