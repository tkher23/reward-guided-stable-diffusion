"""Compute FID (Fréchet Inception Distance) between generated and real images.

Compares three methods by generating images and computing FID vs real product images.

Usage:
  python evals/compute_fid_metrics.py \\
    --csv data/knives_catalog.csv \\
    --image-root data/images \\
    --oracle-checkpoint models/oracle/checkpoints/best.pt \\
    --lora-dir models/diffusion_lora \\
    --num-samples 20 \\
    --output evals/fid_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import transforms
from torchvision.models import inception_v3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NOTE: the jewelry/price generation methods (SteeredDiffusionMethod etc.) are
# imported lazily inside main() so that FIDCalculator can be reused by other
# pipelines (e.g. preference steering) without pulling in the cost-oracle stack.


class FIDCalculator:
    """Compute Fréchet Inception Distance between two sets of images."""

    def __init__(self, device: str = "cuda"):
        self.device = device

        # Load InceptionV3 feature extractor
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # Remove final classification layer
        self.inception.eval()
        self.inception.to(device)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract InceptionV3 features from a PIL image."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.inception(img_tensor)

        return features.cpu().numpy().flatten()

    def compute_fid(self, real_features: np.ndarray, gen_features: np.ndarray) -> float:
        """Compute FID between real and generated feature distributions.

        Args:
            real_features: [N, D] array of real image features
            gen_features: [N, D] array of generated image features

        Returns:
            FID score (lower is better)
        """
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)

        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)

        # Compute Wasserstein distance
        diff = mu_real - mu_gen
        covmean = sqrtm(sigma_real @ sigma_gen)

        # Handle numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = np.sum(diff ** 2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
        return float(fid)


def load_dataset(
    csv_path: str,
    image_root: str,
    prompt_columns: Optional[list] = None,
    price_column: str = "price",
    image_path_column: str = "image_path",
    limit: Optional[int] = None,
    sample_pct: Optional[float] = None,
) -> pd.DataFrame:
    """Load dataset with real images."""
    df = pd.read_csv(csv_path)

    if limit:
        df = df.head(limit)

    if sample_pct is not None:
        frac = max(0.0, min(sample_pct / 100.0, 1.0))
        df = df.sample(frac=frac, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample_pct:.0f}% → {len(df)} rows")

    # Auto-detect prompt columns
    if prompt_columns is None:
        candidates = ["name", "elements", "why_love", "description", "prompt"]
        prompt_columns = [col for col in candidates if col in df.columns]

    if not prompt_columns:
        raise ValueError(f"No prompt columns found. Available: {df.columns.tolist()}")

    # Build prompts
    df["prompt"] = df[prompt_columns].fillna("").agg(" ".join, axis=1).str.strip()

    # Ensure required columns
    if price_column not in df.columns:
        raise ValueError(f"Price column '{price_column}' not found")
    if image_path_column not in df.columns:
        raise ValueError(f"Image path column '{image_path_column}' not found")

    df["target_price"] = df[price_column].astype(float)
    df["image_full_path"] = df[image_path_column].apply(lambda x: os.path.join(image_root, x))

    # Filter to samples with existing images
    df = df[df["image_full_path"].apply(os.path.exists)].reset_index(drop=True)

    print(f"Loaded {len(df)} samples with existing images from {csv_path}")
    print(f"Prompt columns: {prompt_columns}")
    print(f"Price range: ${df['target_price'].min():.0f} - ${df['target_price'].max():.0f}")

    return df[["prompt", "target_price", "image_full_path"]]


def load_real_image(path: str) -> Image.Image:
    """Load and convert real image to RGB."""
    img = Image.open(path).convert("RGB")
    return img


def resolve_oracle(oracle_source: str, oracle_checkpoint: str, hf_model_id: str, hf_token: str) -> str:
    """Download oracle from HF if needed, return local path."""
    if oracle_source == "hf":
        from huggingface_hub import hf_hub_download
        oracle_repo = hf_model_id
        if not oracle_repo:
            raise ValueError("--hf-diffusion-model-id required when --oracle-source hf")
        print(f"Downloading oracle from HuggingFace: {oracle_repo}")
        return hf_hub_download(repo_id=oracle_repo, filename="oracle/best.pt", token=hf_token or None)
    return oracle_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Compute FID metrics for three generation methods")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--image-root", required=True, help="Root directory for images")
    parser.add_argument(
        "--prompt-columns",
        nargs="+",
        default=None,
        help="CSV columns to use for prompts (default: auto-detect)",
    )
    parser.add_argument("--price-column", default="price")
    parser.add_argument("--image-path-column", default="image_path")
    parser.add_argument("--oracle-checkpoint", default="models/oracle/checkpoints/best.pt")
    parser.add_argument("--oracle-source", choices=["local", "hf"], default="local")
    parser.add_argument("--diffusion-source", choices=["hf", "local-finetune"], default="hf")
    parser.add_argument("--hf-diffusion-model-id", default="", help="HF model ID (e.g. BOBALAB/first_draft)")
    parser.add_argument("--hf-token", default="", help="HF token for private repos")
    parser.add_argument("--lora-dir", default="models/diffusion_lora")
    parser.add_argument("--diffusion-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--sample-pct", type=float, default=None, help="Randomly sample X%% of dataset (e.g. 10 for 10%%)")
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--output", default="evals/fid_results.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Generation methods imported here (not at module top) so importing
    # FIDCalculator elsewhere doesn't require the cost-oracle stack.
    from inference.method_steered import SteeredDiffusionMethod
    from inference.method_lora_nosteering import LoRATunedDiffusionNoSteering
    from inference.method_vanilla import VanillaDiffusion

    # Resolve oracle checkpoint
    oracle_ckpt = resolve_oracle(
        args.oracle_source,
        args.oracle_checkpoint,
        args.hf_diffusion_model_id,
        args.hf_token,
    )

    # Load dataset
    df = load_dataset(
        args.csv,
        args.image_root,
        prompt_columns=args.prompt_columns,
        price_column=args.price_column,
        image_path_column=args.image_path_column,
        limit=args.num_samples,
        sample_pct=args.sample_pct,
    )

    # Initialize FID calculator
    print("\n=== Initializing FID Calculator ===")
    fid_calc = FIDCalculator(device=args.device)

    # Initialize three methods
    print("=== Initializing Generation Methods ===")
    method_steered = SteeredDiffusionMethod(
        oracle_checkpoint=oracle_ckpt,
        diffusion_source=args.diffusion_source,
        hf_model_id=args.hf_diffusion_model_id or None,
        diffusion_lora_dir=args.lora_dir,
        diffusion_model_id=args.diffusion_model,
        device=args.device,
    )

    method_lora = LoRATunedDiffusionNoSteering(
        oracle_checkpoint=oracle_ckpt,
        diffusion_source=args.diffusion_source,
        hf_model_id=args.hf_diffusion_model_id or None,
        diffusion_lora_dir=args.lora_dir,
        diffusion_model_id=args.diffusion_model,
        device=args.device,
    )

    method_vanilla = VanillaDiffusion(
        oracle_checkpoint=oracle_ckpt,
        diffusion_model_id=args.diffusion_model,
        device=args.device,
    )

    # Extract features from real images
    print(f"\n=== Extracting Features from {len(df)} Real Images ===")
    real_features = []
    valid_indices = []

    for idx, row in df.iterrows():
        try:
            real_img = load_real_image(row["image_full_path"])
            features = fid_calc.extract_features(real_img)
            real_features.append(features)
            valid_indices.append(idx)
            print(f"  Real image {idx + 1}/{len(df)}: {row['image_full_path']}", flush=True)
        except Exception as e:
            print(f"  Error loading real image {idx}: {e}", flush=True)

    real_features = np.array(real_features)
    df = df.iloc[valid_indices].reset_index(drop=True)
    print(f"Successfully loaded {len(real_features)} real images")

    # Generate images and compute FID
    print(f"\n=== Generating and Computing FID for {len(df)} Samples ===\n")
    print(f"{'Sample':<8} {'Target':>8} {'FID Steered':>12} {'FID LoRA':>10} {'FID Vanilla':>12}")
    print("-" * 58)

    results = {
        "steered": {"fids": []},
        "lora_nosteering": {"fids": []},
        "vanilla": {"fids": []},
    }

    for idx, row in enumerate(df.itertuples(), 1):
        prompt = row.prompt
        target_price = row.target_price

        try:
            # Generate with each method
            img_steered, _ = method_steered.generate(prompt, target_price, args.num_inference_steps)
            img_lora, _ = method_lora.generate(prompt, target_price, args.num_inference_steps)
            img_vanilla, _ = method_vanilla.generate(prompt, target_price, args.num_inference_steps)

            # Extract features
            feat_steered = fid_calc.extract_features(img_steered)
            feat_lora = fid_calc.extract_features(img_lora)
            feat_vanilla = fid_calc.extract_features(img_vanilla)

            # L2 distance between each generated image and its corresponding real image
            fid_steered = float(np.linalg.norm(real_features[idx - 1] - feat_steered))
            fid_lora = float(np.linalg.norm(real_features[idx - 1] - feat_lora))
            fid_vanilla = float(np.linalg.norm(real_features[idx - 1] - feat_vanilla))

            results["steered"]["fids"].append(fid_steered)
            results["lora_nosteering"]["fids"].append(fid_lora)
            results["vanilla"]["fids"].append(fid_vanilla)

            print(
                f"{idx:<8} ${target_price:>7.0f} {fid_steered:>12.4f} {fid_lora:>10.4f} {fid_vanilla:>12.4f}",
                flush=True,
            )

            # Print running averages every 5 samples
            if idx % 5 == 0:
                n = len(results["steered"]["fids"])
                avg_s = sum(results["steered"]["fids"]) / n
                avg_l = sum(results["lora_nosteering"]["fids"]) / n
                avg_v = sum(results["vanilla"]["fids"]) / n
                print(f"\n  >> Running avg FID — Steered: {avg_s:.4f} | LoRA: {avg_l:.4f} | Vanilla: {avg_v:.4f}\n")

        except Exception as e:
            print(f"\nError on sample {idx}: {e}", flush=True)
            continue

    # Compute statistics
    print("\n=== Results ===\n")

    final_results = {
        "num_samples": len(df),
    }

    for method_name in ["steered", "lora_nosteering", "vanilla"]:
        fids = results[method_name]["fids"]
        if fids:
            mean_fid = float(np.mean(fids))
            final_results[method_name] = {
                "mean_fid": mean_fid,
                "median_fid": float(np.median(fids)),
                "min_fid": float(np.min(fids)),
                "max_fid": float(np.max(fids)),
                "std_fid": float(np.std(fids)),
                "fids": fids,
            }

            print(f"{method_name.upper()}:")
            print(f"  Mean FID: {mean_fid:.4f}")
            print(f"  Median FID: {final_results[method_name]['median_fid']:.4f}")
            print(f"  Std FID: {final_results[method_name]['std_fid']:.4f}")
            print()

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
