from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class DifferentiableFeatureExtractor:
    """Extract differentiable CLIP-based features for any image."""

    def __init__(
        self,
        feature_concepts: list[str] = None,
        device: str = "cuda",
        concept_stats_file: str = "sneakerdata/features/concept_stats.json",
    ) -> None:
        if feature_concepts is None:
            feature_concepts = [
                "colorful bright sneaker",
                "chunky platform sneaker",
                "clean minimal sneaker",
            ]

        self.feature_concepts = feature_concepts
        self.device = device
        self.num_features = len(feature_concepts)

        print(f"Loading CLIP model on {device}...")
        # use_safetensors=False loads pytorch_model.bin directly; forcing
        # safetensors triggers transformers' Hub conversion service, which fails
        # on some pods with `KeyError: 'event_id'`.
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", use_safetensors=False
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

        # Load the f(x) transform stats (per-concept mean/std + temperature)
        # written at labeling time, so generated images get the exact same
        # representation as the dataset. Falls back to a plain softmax if absent.
        self.concept_mean = None
        self.concept_std = None
        self.temperature = 1.0
        if concept_stats_file and Path(concept_stats_file).exists():
            with open(concept_stats_file) as f:
                stats = json.load(f)
            self.concept_mean = torch.tensor(stats["mean"], device=device)
            self.concept_std = (
                torch.tensor(stats["std"], device=device).clamp_min(1e-6)
            )
            self.temperature = stats.get("temperature", 1.0)

        # Precompute text feature vectors (static). We go through the
        # text_model + text_projection submodules directly instead of
        # get_text_features, whose return type varies across transformers
        # versions (newer ones return BaseModelOutputWithPooling).
        text_inputs = self.processor(
            text=feature_concepts,
            return_tensors="pt",
            padding=True,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_outputs = self.model.text_model(**text_inputs)
            text_features = self.model.text_projection(text_outputs.pooler_output)
            self.text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

    def _to_features(self, similarities: torch.Tensor) -> torch.Tensor:
        """Map concept cosine similarities -> f(x), matching label time.

        Standardize per concept (debias the per-concept baseline) then softmax
        with the stored temperature. Differentiable.
        """
        if self.concept_mean is not None:
            z = (similarities - self.concept_mean) / self.concept_std
        else:
            z = similarities
        return (z / self.temperature).softmax(dim=-1)

    def extract_features_from_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a normalized image tensor in [0, 1].

        Args:
            image_tensor: [B, 3, H, W] or [3, H, W] in [0, 1]

        Returns:
            [B, num_features] or [num_features] tensor of feature scores in [0, 1]
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Resize to CLIP size (224x224) if needed
        if image_tensor.shape[2] != 224 or image_tensor.shape[3] != 224:
            import torch.nn.functional as F

            image_tensor = F.interpolate(
                image_tensor,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

        image_tensor = image_tensor.to(self.device)

        # Get projected image features (differentiable) via the vision tower
        # and visual projection directly.
        with torch.enable_grad():
            vision_outputs = self.model.vision_model(pixel_values=image_tensor)
            image_features = self.model.visual_projection(vision_outputs.pooler_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # f(x) over the concepts (matches labeling). Differentiable, so
            # gradients flow back to the image for steering.
            similarities = image_features @ self.text_features.T  # [B, num_features]
            features = self._to_features(similarities)

        return features.squeeze(0) if features.shape[0] == 1 else features

    def extract_features_from_pil(self, image: Image.Image) -> torch.Tensor:
        """Extract features from a PIL image."""
        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            image_features = self.model.visual_projection(vision_outputs.pooler_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarities = image_features @ self.text_features.T
        features = self._to_features(similarities)

        return features.squeeze(0)


def validate_feature_space(
    image_dir: str = "sneakerdata/images/shoes_100",
    labels_file: str = "sneakerdata/features/feature_labels.json",
) -> None:
    """
    Validate that CLIP space captures the intended features.
    Compare AI-labeled discrete labels with continuous CLIP scores.
    """
    extractor = DifferentiableFeatureExtractor()

    with open(labels_file) as f:
        labels = json.load(f)

    print("\nValidating feature space...")
    print("Image ID | Colorful Label | Colorful Score | Chunky Label | Chunky Score | Clean Label | Clean Score")
    print("-" * 120)

    for img_id, label_data in list(labels.items())[:10]:
        img_path = label_data["image_path"]
        image = Image.open(img_path).convert("RGB")
        features = extractor.extract_features_from_pil(image)

        colorful_label = "HIGH" if label_data["colorful"] > 0.5 else "LOW"
        chunky_label = "HIGH" if label_data["chunky"] > 0.5 else "LOW"
        clean_label = "HIGH" if label_data["clean"] > 0.5 else "LOW"

        print(
            f"{img_id:10} | {colorful_label:14} | {features[0].item():14.3f} | "
            f"{chunky_label:12} | {features[1].item():12.3f} | {clean_label:11} | {features[2].item():11.3f}"
        )

    print("Validation complete!")


if __name__ == "__main__":
    validate_feature_space()
