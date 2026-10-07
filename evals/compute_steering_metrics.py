"""Compute steering success metrics across three methods.

Generates images with three methods (steered, LoRA-no-steer, vanilla) and compares
oracle price predictions against target prices.

Usage:
  python evals/compute_steering_metrics.py \\
    --csv data/knives_catalog.csv \\
    --oracle-checkpoint models/oracle/checkpoints/best.pt \\
    --lora-dir models/diffusion_lora \\
    --output eval/steering_results.json \\
    --num-samples 20
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.method_steered import SteeredDiffusionMethod
from inference.method_lora_nosteering import LoRATunedDiffusionNoSteering
from inference.method_vanilla import VanillaDiffusion


def load_dataset(
    csv_path: str,
    prompt_columns: Optional[list] = None,
    price_column: str = "price",
    limit: Optional[int] = None,
    sample_pct: Optional[float] = None,
) -> pd.DataFrame:
    """Load dataset from CSV.

    Args:
        csv_path: path to CSV file
        prompt_columns: which columns to concatenate for prompt text
                       If None, infers from data (tries: name, elements, why_love)
        price_column: name of price column
        limit: max number of rows to load
        sample_pct: randomly sample this percentage of rows (0-100), applied after limit

    Returns: DataFrame with columns 'prompt' and 'target_price'
    """
    df = pd.read_csv(csv_path)

    if limit:
        df = df.head(limit)

    if sample_pct is not None:
        frac = max(0.0, min(sample_pct / 100.0, 1.0))
        df = df.sample(frac=frac, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample_pct:.0f}% → {len(df)} rows")

    # Auto-detect prompt columns if not specified
    if prompt_columns is None:
        candidates = ["name", "elements", "why_love", "description", "prompt"]
        prompt_columns = [col for col in candidates if col in df.columns]

    if not prompt_columns:
        raise ValueError(
            f"No prompt columns found. Available: {df.columns.tolist()}. "
            "Specify --prompt-columns."
        )

    # Concatenate selected columns into prompt text
    df["prompt"] = df[prompt_columns].fillna("").agg(" ".join, axis=1).str.strip()

    # Ensure price column exists
    if price_column not in df.columns:
        raise ValueError(f"Price column '{price_column}' not found in CSV. Available: {df.columns.tolist()}")

    df["target_price"] = df[price_column].astype(float)

    # Keep only needed columns
    df = df[["prompt", "target_price"]].reset_index(drop=True)

    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Prompt columns: {prompt_columns}")
    print(f"Price range: ${df['target_price'].min():.0f} - ${df['target_price'].max():.0f}")

    return df


def compute_metrics(errors: list) -> dict:
    """Compute statistics on price errors."""
    errors = [abs(e) for e in errors]  # absolute errors

    return {
        "mean_error": float(sum(errors) / len(errors)),
        "median_error": float(sorted(errors)[len(errors) // 2]),
        "min_error": float(min(errors)),
        "max_error": float(max(errors)),
        "std_error": float((sum((x - sum(errors) / len(errors)) ** 2 for x in errors) / len(errors)) ** 0.5),
        "pct_within_15": float(sum(1 for e in errors if e <= 15) / len(errors) * 100),
        "pct_within_25": float(sum(1 for e in errors if e <= 25) / len(errors) * 100),
        "pct_within_50": float(sum(1 for e in errors if e <= 50) / len(errors) * 100),
    }


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
    parser = argparse.ArgumentParser(description="Compute steering success metrics")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument(
        "--prompt-columns",
        nargs="+",
        default=None,
        help="CSV columns to use for prompts (default: auto-detect name/elements/why_love)",
    )
    parser.add_argument("--price-column", default="price", help="CSV column with prices")
    parser.add_argument("--oracle-checkpoint", default="models/oracle/checkpoints/best.pt")
    parser.add_argument("--oracle-source", choices=["local", "hf"], default="local")
    parser.add_argument("--diffusion-source", choices=["hf", "local-finetune"], default="hf")
    parser.add_argument("--hf-diffusion-model-id", default="", help="HF model ID (e.g. BOBALAB/first_draft)")
    parser.add_argument("--hf-token", default="", help="HF token for private repos")
    parser.add_argument("--lora-dir", default="models/diffusion_lora", help="Local LoRA dir (local-finetune only)")
    parser.add_argument("--diffusion-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num-samples", type=int, default=None, help="Max samples to test (absolute count)")
    parser.add_argument("--sample-pct", type=float, default=None, help="Randomly sample X%% of dataset (e.g. 10 for 10%%)")
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--output", default="evals/steering_results.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

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
        prompt_columns=args.prompt_columns,
        price_column=args.price_column,
        limit=args.num_samples,
        sample_pct=args.sample_pct,
    )

    # Initialize three methods
    print("\n=== Initializing Methods ===")
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

    # Generate and measure
    print(f"\n=== Generating {len(df)} images with 3 methods ===\n")
    print(f"{'Sample':<8} {'Target':>8} {'Steered':>10} {'Err':>8} {'LoRA':>10} {'Err':>8} {'Vanilla':>10} {'Err':>8}")
    print("-" * 75)

    results = {
        "steered": {"errors": [], "predictions": []},
        "lora_nosteering": {"errors": [], "predictions": []},
        "vanilla": {"errors": [], "predictions": []},
    }

    for idx, row in enumerate(df.itertuples(), 1):
        prompt = row.prompt
        target_price = row.target_price

        try:
            _, pred_steered = method_steered.generate(prompt, target_price, args.num_inference_steps)
            _, pred_lora = method_lora.generate(prompt, target_price, args.num_inference_steps)
            _, pred_vanilla = method_vanilla.generate(prompt, target_price, args.num_inference_steps)

            err_steered = pred_steered - target_price
            err_lora = pred_lora - target_price
            err_vanilla = pred_vanilla - target_price

            results["steered"]["predictions"].append(pred_steered)
            results["steered"]["errors"].append(err_steered)
            results["lora_nosteering"]["predictions"].append(pred_lora)
            results["lora_nosteering"]["errors"].append(err_lora)
            results["vanilla"]["predictions"].append(pred_vanilla)
            results["vanilla"]["errors"].append(err_vanilla)

            # Print per-sample result immediately
            print(
                f"{idx:<8} ${target_price:>7.0f} ${pred_steered:>9.2f} {err_steered:>+8.2f} "
                f"${pred_lora:>9.2f} {err_lora:>+8.2f} "
                f"${pred_vanilla:>9.2f} {err_vanilla:>+8.2f}",
                flush=True,
            )

            # Print running averages every 5 samples
            if idx % 5 == 0:
                n = len(results["steered"]["errors"])
                avg_s = sum(abs(e) for e in results["steered"]["errors"]) / n
                avg_l = sum(abs(e) for e in results["lora_nosteering"]["errors"]) / n
                avg_v = sum(abs(e) for e in results["vanilla"]["errors"]) / n
                print(f"\n  >> Running avg error — Steered: ${avg_s:.2f} | LoRA: ${avg_l:.2f} | Vanilla: ${avg_v:.2f}\n")

        except Exception as e:
            print(f"\nError on sample {idx}: {e}", flush=True)
            continue

    # Compute metrics for each method
    print("\n=== Results ===\n")

    final_results = {
        "num_samples": len(results["steered"]["errors"]),
        "target_prices": df["target_price"].tolist(),
    }

    for method_name in ["steered", "lora_nosteering", "vanilla"]:
        errors = results[method_name]["errors"]
        if errors:
            metrics = compute_metrics(errors)
            final_results[method_name] = metrics

            print(f"{method_name.upper()}:")
            print(f"  Mean Error: ${metrics['mean_error']:.2f}")
            print(f"  Median Error: ${metrics['median_error']:.2f}")
            print(f"  % within ±$15: {metrics['pct_within_15']:.1f}%")
            print(f"  % within ±$25: {metrics['pct_within_25']:.1f}%")
            print()

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
