from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def label_features_with_clip(
    image_dir: str = "sneakerdata/images/shoes_100",
    output_dir: str = "sneakerdata/features",
    feature_concepts: list[str] = None,
    temperature: float = 1.0,
) -> dict:
    """
    Use CLIP zero-shot to label shoe images with discrete features.

    Returns mapping: {image_id: {"colorful": score, "chunky": score, "clean": score}}
    """
    if feature_concepts is None:
        feature_concepts = [
            "colorful bright sneaker",
            "chunky platform sneaker",
            "clean minimal sneaker",
        ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")

    # use_safetensors=False loads the model's pytorch_model.bin directly. Forcing
    # safetensors triggers transformers' on-the-fly Hub conversion service, which
    # fails on some pods with `KeyError: 'event_id'` before any fallback.
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", use_safetensors=False
    ).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_paths = sorted(Path(image_dir).glob("*.jpg"))
    print(f"Found {len(image_paths)} images to label")

    # Pass 1: raw cosine similarity of each image against each concept.
    ids, paths, cos_rows = [], [], []
    embeddings = {}

    for i, img_path in enumerate(image_paths):
        if i % 10 == 0:
            print(f"Processing {i}/{len(image_paths)}")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        # Encode image + text concepts in a single forward pass and read the
        # projected `image_embeds`/`text_embeds` off the output. We avoid
        # get_image_features/get_text_features because their return type
        # varies across transformers versions (newer ones return the raw
        # BaseModelOutputWithPooling instead of a projected tensor).
        inputs = processor(
            text=feature_concepts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        image_features = outputs.image_embeds  # (1, D)
        text_features = outputs.text_embeds  # (num_concepts, D)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        cos = (image_features @ text_features.T).squeeze(0)  # (num_concepts,)

        img_id = img_path.stem
        ids.append(img_id)
        paths.append(str(img_path))
        cos_rows.append(cos.cpu())
        # Full L2-normalized CLIP image embedding for the "embedding" feature
        # space.
        embeddings[img_id] = image_features.squeeze(0).cpu().tolist()

    # Pass 2: turn raw cosines into f(x). Raw CLIP cosines cluster tightly
    # (~0.2-0.3) AND one concept ("chunky platform sneaker") sits higher for
    # *every* shoe, so a plain softmax collapses every image onto that concept.
    # We first standardize each concept across the dataset (removing that
    # per-concept baseline) and then softmax with a mild temperature, giving a
    # spread, diverse distribution per image, e.g. [0.6, 0.3, 0.1].
    cos_mat = torch.stack(cos_rows)  # [N, num_concepts]
    mean = cos_mat.mean(dim=0)
    std = cos_mat.std(dim=0).clamp_min(1e-6)
    z = (cos_mat - mean) / std
    fx = (z / temperature).softmax(dim=-1)  # [N, num_concepts]

    labels = {}
    for idx, img_id in enumerate(ids):
        scores = fx[idx].tolist()
        labels[img_id] = {
            # Ordered to match `feature_concepts`; the loader uses this list.
            "concept_scores": scores,
            # Named keys kept for human-readable inspection (assume the default
            # colorful/chunky/clean concept order).
            "colorful": scores[0],
            "chunky": scores[1] if len(scores) > 1 else None,
            "clean": scores[2] if len(scores) > 2 else None,
            "image_path": paths[idx],
        }

    # Save concept labels
    output_file = output_path / "feature_labels.json"
    with open(output_file, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Labels saved to {output_file}")

    # Save the f(x) transform stats so steering-time feature extraction can
    # reproduce the exact same representation on generated images.
    stats_file = output_path / "concept_stats.json"
    with open(stats_file, "w") as f:
        json.dump(
            {
                "concepts": feature_concepts,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "temperature": temperature,
            },
            f,
            indent=2,
        )
    print(f"Concept stats saved to {stats_file}")

    # Save raw image embeddings (used by the "embedding" feature space)
    embeddings_file = output_path / "image_embeddings.json"
    with open(embeddings_file, "w") as f:
        json.dump(embeddings, f)
    print(f"Embeddings saved to {embeddings_file}")

    return labels


if __name__ == "__main__":
    label_features_with_clip()
