from __future__ import annotations

import os

# Force transformers to skip TensorFlow backend imports in this environment.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from typing import Dict, Tuple
from PIL import Image
import numpy as np
from transformers import CLIPModel, CLIPProcessor


_REFERENCE_CACHE: dict[str, tuple[CLIPModel, CLIPProcessor]] = {}

# Simple keyword groups for prompt-attribute diagnostics.
ATTRIBUTE_GROUPS = {
    "material": ["silver", "gold", "rose gold", "platinum", "sterling"],
    "color": [
        "blue",
        "green",
        "red",
        "black",
        "white",
        "pink",
        "purple",
        "yellow",
    ],
    "gem": ["diamond", "pearl", "ruby", "sapphire", "emerald", "stone", "crystal"],
    "type": ["ring", "bracelet", "necklace", "earring", "pendant", "charm"],
}


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def _ensure_pil_image(img) -> Image.Image:
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype("uint8")
        return Image.fromarray(img)
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(img)


def _extract_attributes(text: str) -> list[str]:
    lower = (text or "").lower()
    found = []
    for _, words in ATTRIBUTE_GROUPS.items():
        for w in words:
            if w in lower:
                found.append(w)
    return found


def _compute_margin_metrics(similarity: torch.Tensor) -> Dict[str, float]:
    """Compute diagonal-vs-negative margin metrics from a similarity matrix."""
    if similarity.numel() == 0:
        return {
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "hard_negative_mean": 0.0,
        }

    n = min(similarity.shape[0], similarity.shape[1])
    if n == 0:
        return {
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "hard_negative_mean": 0.0,
        }

    margins = []
    hard_negs = []
    for i in range(n):
        pos = similarity[i, i]
        row = similarity[i]
        if row.shape[0] <= 1:
            hard_neg = torch.tensor(0.0, device=row.device)
        else:
            mask = torch.ones_like(row, dtype=torch.bool)
            mask[i] = False
            hard_neg = row[mask].max()
        margins.append((pos - hard_neg).item())
        hard_negs.append(hard_neg.item())

    margins_np = np.array(margins, dtype=np.float32)
    hard_np = np.array(hard_negs, dtype=np.float32)
    return {
        "mean_margin": float(margins_np.mean()),
        "min_margin": float(margins_np.min()),
        "hard_negative_mean": float(hard_np.mean()),
    }


@torch.no_grad()
def _get_reference_clip(
    reference_model_name: str,
    device: str,
) -> tuple[CLIPModel, CLIPProcessor]:
    key = f"{reference_model_name}@{device}"
    if key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]

    processor = CLIPProcessor.from_pretrained(reference_model_name)
    model = CLIPModel.from_pretrained(reference_model_name).to(device)
    model.eval()
    _REFERENCE_CACHE[key] = (model, processor)
    return model, processor


@torch.no_grad()
def encode_with_reference_clip(
    images: list[Image.Image],
    texts: list[str],
    device: str,
    reference_model_name: str = "openai/clip-vit-large-patch14",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode images/texts with a stronger frozen CLIP reference model."""
    if len(images) == 0 or len(texts) == 0:
        return torch.empty(0), torch.empty(0)

    model, processor = _get_reference_clip(reference_model_name, device)
    pil_images = [_ensure_pil_image(img) for img in images]

    image_inputs = processor(images=pil_images, return_tensors="pt")
    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)

    image_feats = model.get_image_features(
        pixel_values=image_inputs["pixel_values"].to(device)
    )
    text_feats = model.get_text_features(
        input_ids=text_inputs["input_ids"].to(device),
        attention_mask=text_inputs["attention_mask"].to(device),
    )
    return _normalize(image_feats).cpu(), _normalize(text_feats).cpu()


def encode_generated_images_and_texts(
    model_clip,
    images: list[Image.Image],
    texts: list[str],
    processor,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode generated images and their corresponding text prompts using trained CLIP.
    
    Args:
        model_clip: Trained CLIPFinetune model
        images: List of PIL Image objects (generated designs)
        texts: List of text prompts that generated those images
        processor: CLIPProcessor
        device: Device (cuda/cpu)
    
    Returns:
        Tuple of (image_embeddings, text_embeddings) both normalized
    """
    image_embeddings = []
    text_embeddings = []

    with torch.no_grad():
        # Process images in batches
        for img in images:
            img = _ensure_pil_image(img)

            processed = processor(images=[img], return_tensors="pt")
            pixel_values = processed["pixel_values"].to(device)
            img_embed = model_clip.get_image_embeddings(pixel_values)
            image_embeddings.append(img_embed.cpu())

        # Process texts in batches
        for text in texts:
            processed = processor(text=[text], return_tensors="pt", padding=True)
            input_ids = processed["input_ids"].to(device)
            attention_mask = processed["attention_mask"].to(device)
            text_embed = model_clip.get_text_embeddings(input_ids, attention_mask)
            text_embeddings.append(text_embed.cpu())

    image_embeddings = _normalize(torch.cat(image_embeddings, dim=0))
    text_embeddings = _normalize(torch.cat(text_embeddings, dim=0))

    return image_embeddings, text_embeddings


@torch.no_grad()
def evaluate_generation_quality(
    model_clip,
    images: list[Image.Image],
    texts: list[str],
    processor,
    device: str,
) -> Dict[str, float]:
    """
    Evaluate how well generated images match their text prompts using trained CLIP.
    
    Returns metrics:
    - mean_similarity: Average cosine similarity between matched image-text pairs
    - binding_score: How well attributes are "bound" (variance in similarities)
    """
    if len(images) == 0 or len(texts) == 0:
        return {
            "mean_similarity": 0.0,
            "binding_score": 0.0,
            "min_similarity": 0.0,
            "max_similarity": 0.0,
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "hard_negative_mean": 0.0,
        }

    image_embeddings, text_embeddings = encode_generated_images_and_texts(
        model_clip, images, texts, processor, device
    )

    similarity_matrix = text_embeddings @ image_embeddings.T

    # Matched diagonal similarities
    similarities = []
    for i in range(min(len(image_embeddings), len(text_embeddings))):
        sim = similarity_matrix[i, i].item()
        similarities.append(sim)

    similarities = np.array(similarities)
    mean_sim = float(np.mean(similarities))
    binding_score = float(np.std(similarities))  # Lower std = more consistent binding

    margin_metrics = _compute_margin_metrics(similarity_matrix)

    return {
        "mean_similarity": mean_sim,
        "binding_score": binding_score,  # Higher = less consistent (bad)
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        **margin_metrics,
    }


@torch.no_grad()
def evaluate_retrieval_ranking(
    model_clip,
    generated_images: list[Image.Image],
    query_texts: list[str],
    processor,
    device: str,
    topk: int = 5,
) -> Dict[str, float]:
    """
    Evaluate if correct image ranks high for its intended text query.
    Simulates: given text, does the model's generated image rank at top?
    """
    if len(generated_images) == 0 or len(query_texts) == 0:
        return {"top_k_accuracy": 0.0}

    image_embeddings, text_embeddings = encode_generated_images_and_texts(
        model_clip, generated_images, query_texts, processor, device
    )

    # Compute similarity matrix between all texts and images
    similarity = text_embeddings @ image_embeddings.T  # [num_texts, num_images]

    # For each text, check if matching image ranks in top-k
    correct = 0
    for i in range(len(query_texts)):
        ranked = similarity[i].argsort(descending=True)
        if i in ranked[:topk]:
            correct += 1

    return {"top_k_accuracy": correct / len(query_texts)}


@torch.no_grad()
def evaluate_reference_clip_quality(
    images: list[Image.Image],
    texts: list[str],
    device: str,
    reference_model_name: str = "openai/clip-vit-large-patch14",
) -> Dict[str, float]:
    """Quality metrics computed with a stronger frozen CLIP reference model."""
    if len(images) == 0 or len(texts) == 0:
        return {
            "reference_mean_similarity": 0.0,
            "reference_mean_margin": 0.0,
            "reference_top1": 0.0,
        }

    image_embeddings, text_embeddings = encode_with_reference_clip(
        images=images,
        texts=texts,
        device=device,
        reference_model_name=reference_model_name,
    )
    similarity = text_embeddings @ image_embeddings.T
    n = min(similarity.shape[0], similarity.shape[1])
    diag = [similarity[i, i].item() for i in range(n)] if n > 0 else [0.0]
    margin_metrics = _compute_margin_metrics(similarity)

    top1_correct = 0
    for i in range(n):
        ranked = similarity[i].argsort(descending=True)
        if int(ranked[0]) == i:
            top1_correct += 1

    return {
        "reference_mean_similarity": float(np.mean(np.array(diag, dtype=np.float32))),
        "reference_mean_margin": margin_metrics["mean_margin"],
        "reference_top1": float(top1_correct / max(n, 1)),
    }


@torch.no_grad()
def evaluate_attribute_binding(
    images: list[Image.Image],
    texts: list[str],
    device: str,
    reference_model_name: str = "openai/clip-vit-large-patch14",
) -> Dict[str, float]:
    """Estimate whether prompt attributes are jointly present in generated images."""
    if len(images) == 0 or len(texts) == 0:
        return {
            "attribute_presence_score": 0.0,
            "attribute_margin_score": 0.0,
            "avg_attributes_per_prompt": 0.0,
        }

    model, processor = _get_reference_clip(reference_model_name, device)

    pil_images = [_ensure_pil_image(img) for img in images]
    image_inputs = processor(images=pil_images, return_tensors="pt")
    image_embeds = model.get_image_features(
        pixel_values=image_inputs["pixel_values"].to(device)
    )
    image_embeds = _normalize(image_embeds)

    total_presence = []
    total_margin = []
    attr_counts = []

    for i, prompt in enumerate(texts[: len(pil_images)]):
        attrs = _extract_attributes(prompt)
        attr_counts.append(len(attrs))
        if not attrs:
            total_presence.append(0.0)
            total_margin.append(0.0)
            continue

        pos_texts = [f"jewelry with {a}" for a in attrs]
        neg_texts = ["jewelry"]

        text_inputs = processor(text=pos_texts + neg_texts, return_tensors="pt", padding=True, truncation=True)
        text_embeds = model.get_text_features(
            input_ids=text_inputs["input_ids"].to(device),
            attention_mask=text_inputs["attention_mask"].to(device),
        )
        text_embeds = _normalize(text_embeds)

        img = image_embeds[i : i + 1]
        pos_scores = (img @ text_embeds[: len(pos_texts)].T).squeeze(0)
        neg_score = (img @ text_embeds[len(pos_texts) :].T).squeeze(0).mean()

        presence = float(pos_scores.mean().item())
        margin = float((pos_scores.mean() - neg_score).item())
        total_presence.append(presence)
        total_margin.append(margin)

    return {
        "attribute_presence_score": float(np.mean(np.array(total_presence, dtype=np.float32))),
        "attribute_margin_score": float(np.mean(np.array(total_margin, dtype=np.float32))),
        "avg_attributes_per_prompt": float(np.mean(np.array(attr_counts, dtype=np.float32))),
    }


@torch.no_grad()
def evaluate_generation_suite(
    model_clip,
    images: list[Image.Image],
    texts: list[str],
    processor,
    device: str,
    topk: int = 5,
    reference_model_name: str = "openai/clip-vit-large-patch14",
) -> Dict[str, float]:
    """Full evaluation suite combining trained CLIP + reference CLIP + binding diagnostics."""
    trained_metrics = evaluate_generation_quality(
        model_clip=model_clip,
        images=images,
        texts=texts,
        processor=processor,
        device=device,
    )
    retrieval_metrics = evaluate_retrieval_ranking(
        model_clip=model_clip,
        generated_images=images,
        query_texts=texts,
        processor=processor,
        device=device,
        topk=topk,
    )
    reference_metrics = evaluate_reference_clip_quality(
        images=images,
        texts=texts,
        device=device,
        reference_model_name=reference_model_name,
    )
    binding_metrics = evaluate_attribute_binding(
        images=images,
        texts=texts,
        device=device,
        reference_model_name=reference_model_name,
    )

    return {
        **trained_metrics,
        **retrieval_metrics,
        **reference_metrics,
        **binding_metrics,
    }
