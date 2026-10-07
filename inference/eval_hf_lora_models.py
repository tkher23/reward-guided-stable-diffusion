from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import torch
from huggingface_hub import HfApi, snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.diffusion_steering import CostSteeredGenerator, SteeringConfig
from inference.oracle_inference import OracleInference
from models.model_diffusion import DiffusionGeneratorLoRA

DEFAULT_PROMPTS = [
    "minimal silver bracelet with blue gemstone accents, studio lighting",
    "elegant gold bracelet with heart charm, luxury product photo",
    "rose gold chain bracelet with pearl details, clean white background",
    "modern geometric bracelet in matte silver, high-end catalog style",
    "bohemian bracelet with turquoise stones and delicate engravings",
]


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _read_prompts(prompt_file: str | None) -> list[str]:
    if not prompt_file:
        return DEFAULT_PROMPTS

    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_file}")

    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) < 5:
        raise ValueError("Prompt file must contain at least 5 non-empty prompts (one per line).")
    return prompts[:5]


def _discover_lora_dirs(repo_id: str, token: str | None, revision: str | None) -> list[str]:
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)

    found = set()
    for file_path in files:
        if file_path.endswith("adapter_config.json"):
            found.add(str(Path(file_path).parent).replace("\\", "/"))

    if not found:
        raise RuntimeError(
            "No LoRA adapters found in repo. Expected folders containing adapter_config.json files."
        )

    return sorted(found)


def _download_lora_dirs(
    repo_id: str,
    lora_dirs: Iterable[str],
    cache_root: Path,
    token: str | None,
    revision: str | None,
) -> dict[str, Path]:
    repo_cache_root = cache_root / _safe_name(repo_id)
    repo_cache_root.mkdir(parents=True, exist_ok=True)

    local_paths: dict[str, Path] = {}
    for rel_dir in lora_dirs:
        local_root = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            token=token,
            local_dir=str(repo_cache_root),
            local_dir_use_symlinks=False,
            allow_patterns=[f"{rel_dir}/*"],
        )
        local_paths[rel_dir] = Path(local_root) / rel_dir
    return local_paths


def _build_model_label_map(lora_dirs: list[str]) -> dict[str, str]:
    label_map: dict[str, str] = {}
    for idx, rel_dir in enumerate(lora_dirs, start=1):
        label_map[rel_dir] = f"Model {idx} (m{idx})"
    return label_map


def _save_image_if_needed(image, save_images: bool, save_path: Path) -> None:
    if not save_images:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def run_eval(
    hf_repo_id: str,
    oracle_checkpoint: str,
    base_model_id: str,
    target_price: float,
    steps: int,
    guidance_scale: float,
    steering_strength: float,
    steer_every_k_steps: int,
    save_images: bool,
    output_dir: str,
    hf_token: str | None,
    revision: str | None,
    prompt_file: str | None,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = _read_prompts(prompt_file)

    print(f"Using device: {device}")
    print(f"Reading LoRA checkpoints from HF repo: {hf_repo_id}")
    lora_dirs = _discover_lora_dirs(hf_repo_id, hf_token, revision)
    label_map = _build_model_label_map(lora_dirs)

    print("Discovered adapters:")
    for rel_dir in lora_dirs:
        print(f"  - {label_map[rel_dir]} -> {rel_dir}")

    local_cache_root = PROJECT_ROOT / "models" / "hf_lora_cache"
    print(f"Downloading adapters into: {local_cache_root}")
    local_paths = _download_lora_dirs(
        repo_id=hf_repo_id,
        lora_dirs=lora_dirs,
        cache_root=local_cache_root,
        token=hf_token,
        revision=revision,
    )

    oracle = OracleInference(oracle_checkpoint, device=device)
    output_root = (PROJECT_ROOT / output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for rel_dir in lora_dirs:
        model_label = label_map[rel_dir]
        lora_local_dir = local_paths[rel_dir]

        print("\n" + "=" * 80)
        print(f"Running {model_label} using adapter: {lora_local_dir}")
        print("=" * 80)

        diffusion = DiffusionGeneratorLoRA(model_id=base_model_id, device=device)
        diffusion.setup_lora()
        diffusion.load_lora(str(lora_local_dir))

        steered = CostSteeredGenerator(
            pipeline=diffusion.pipeline,
            oracle=oracle,
            device=device,
        )

        steer_cfg = SteeringConfig(
            target_price=float(target_price),
            steering_strength=float(steering_strength),
            steer_every_k_steps=int(steer_every_k_steps),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(steps),
        )

        for prompt_idx, prompt in enumerate(prompts, start=1):
            image = steered.generate(prompt=prompt, config=steer_cfg)
            pred_price = float(oracle.score_pil_images([image])[0].item())
            abs_error = abs(pred_price - float(target_price))

            print(
                f"[{model_label}] prompt {prompt_idx}/5 | "
                f"predicted={pred_price:.2f} | target={target_price:.2f} | abs_error={abs_error:.2f}"
            )

            image_name = f"prompt_{prompt_idx}.png"
            model_dir = output_root / _safe_name(model_label)
            _save_image_if_needed(image, save_images, model_dir / image_name)

            all_results.append(
                {
                    "model_label": model_label,
                    "model_source_dir": rel_dir,
                    "prompt_index": prompt_idx,
                    "prompt_text": prompt,
                    "predicted_price": pred_price,
                    "target_price": float(target_price),
                    "abs_error": abs_error,
                    "saved_image": str((model_dir / image_name).relative_to(PROJECT_ROOT)) if save_images else "",
                }
            )

    results_path = output_root / "hf_multi_model_eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved metrics:")
    print(f"  {results_path}")
    if save_images:
        print(f"Saved images under: {output_root}")
    else:
        print("Image saving disabled. Use --save-images to enable image export.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download all LoRA adapters from a Hugging Face model repo, label them as Model 1/Model 2/... "
            "and run 5-prompt steered inference with oracle guidance."
        )
    )
    parser.add_argument("--hf-repo-id", type=str, required=True, help="HF model repo that contains multiple LoRA folders")
    parser.add_argument("--oracle-checkpoint", type=str, default="models/oracle/checkpoints/best.pt", help="Path to local oracle checkpoint")
    parser.add_argument("--base-model-id", type=str, default="runwayml/stable-diffusion-v1-5", help="Base diffusion model used with adapters")
    parser.add_argument("--target-price", type=float, required=True, help="Target budget for steering")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--steering-strength", type=float, default=0.15, help="Oracle steering strength")
    parser.add_argument("--steer-every-k-steps", type=int, default=2, help="Apply steering every k denoising steps")
    parser.add_argument("--save-images", action="store_true", help="If set, save generated images for model comparison")
    parser.add_argument("--output-dir", type=str, default="inference/hf_model_comparison", help="Where metrics and optional images are written")
    parser.add_argument("--hf-token", type=str, default=None, help="HF token if the repo is private")
    parser.add_argument("--revision", type=str, default=None, help="Optional branch, tag, or commit")
    parser.add_argument("--prompt-file", type=str, default=None, help="Optional text file with prompts (one per line, first 5 used)")
    args = parser.parse_args()

    run_eval(
        hf_repo_id=args.hf_repo_id,
        oracle_checkpoint=args.oracle_checkpoint,
        base_model_id=args.base_model_id,
        target_price=args.target_price,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        steering_strength=args.steering_strength,
        steer_every_k_steps=args.steer_every_k_steps,
        save_images=args.save_images,
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        revision=args.revision,
        prompt_file=args.prompt_file,
    )


if __name__ == "__main__":
    main()
