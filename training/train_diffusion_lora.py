# python training/train_diffusion_lora.py

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Force transformers/diffusers to avoid TensorFlow backend in this environment.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_diffusion import DiffusionGeneratorLoRA


@dataclass
class DiffusionTrainConfig:
    csv_path: str = "data/catalog_clip_ready.csv"
    image_root: str = "data/images"
    output_dir: str = "models/diffusion_lora"
    clip_checkpoint: str = "models/clip_baseline/checkpoints/best.pt"

    diffusion_model_id: str = "runwayml/stable-diffusion-v1-5"
    cpu_fallback_model_id: str = "runwayml/stable-diffusion-v1-5"
    batch_size: int = 4
    num_epochs: int = 5
    lr: float = 1e-5
    lora_rank: int = 64
    lora_alpha: int = 128

    seed: int = 42
    bracelets_only: bool = True


def resolve_existing_path(path_str: str) -> str:
    p = Path(path_str)
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, PROJECT_ROOT / p]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return ""


def resolve_csv_path(csv_path: str) -> str:
    resolved = resolve_existing_path(csv_path)
    if resolved:
        return resolved

    # Common case: user passes "catalog_clip_ready.csv" while file lives under data/.
    direct_data_candidate = PROJECT_ROOT / "data" / csv_path
    if direct_data_candidate.exists():
        return str(direct_data_candidate.resolve())

    fallback_candidates = [
        PROJECT_ROOT / "data" / "all_products.csv",
        PROJECT_ROOT / "data" / "all_products_partial.csv",
    ]
    for c in fallback_candidates:
        if c.exists():
            print(
                f"CSV '{csv_path}' not found. Falling back to '{c.as_posix()}'."
            )
            return str(c.resolve())

    data_dir = PROJECT_ROOT / "data"
    available = []
    if data_dir.exists():
        available = sorted([p.name for p in data_dir.glob("*.csv")])
    raise FileNotFoundError(
        "Could not resolve CSV path. Tried current working directory and repo root. "
        f"Requested: '{csv_path}'. Available CSVs in data/: {available}"
    )


def resolve_output_dir(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


class JewelryPromptDataset(Dataset):
    """Dataset that yields (prompt, pixel_values) pairs for diffusion training."""

    def __init__(self, csv_path: str, image_root: str, bracelets_only: bool = True):
        self.df = pd.read_csv(csv_path)
        if bracelets_only and "image_path" in self.df.columns:
            self.df = self.df[self.df["image_path"].astype(str).str.startswith("bracelets/")].reset_index(drop=True)

        if "image_path" not in self.df.columns:
            raise ValueError(
                "CSV must contain an 'image_path' column for training. "
                "Use data/catalog_clip_ready.csv (or pass --csv data/catalog_clip_ready.csv)."
            )

        self.image_root = Path(image_root)
        self.transform = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        # Build a robust prompt column once so we don't depend on one specific CSV schema.
        self.df["prompt"] = self.df.apply(self._build_prompt, axis=1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_rel = str(row.get("image_path", "")).strip()
        image_path = self.image_root / image_rel
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        return {"prompt": row["prompt"], "pixel_values": pixel_values}

    @staticmethod
    def _build_prompt(row: pd.Series) -> str:
        name = str(row.get("product_name") or row.get("name") or "jewelry piece").strip()
        elements = str(row.get("elements") or "").strip()
        why_love = str(row.get("why_youll_love_it") or row.get("why_love") or row.get("description") or "").strip()

        parts = [name]
        if elements:
            parts.append(f"Elements: {elements}")
        if why_love:
            parts.append(why_love)
        return ". ".join(parts)


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def train_diffusion_lora(config: DiffusionTrainConfig) -> None:
    """Minimal but real LoRA fine-tuning loop over all dataset images."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weight_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}")

    csv_path = resolve_csv_path(config.csv_path)
    output_dir = resolve_output_dir(config.output_dir)

    print(f"Resolved csv path: {csv_path}")
    print(f"Resolved output dir: {output_dir}")

    # Create output dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "checkpoints").mkdir(parents=True, exist_ok=True)

    # Load diffusion model with LoRA
    print("Loading SDXL with LoRA...")
    diffusion_model = DiffusionGeneratorLoRA(
        model_id=config.diffusion_model_id,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        device=device,
        cpu_fallback_model_id=config.cpu_fallback_model_id,
    )
    diffusion_model.setup_lora()

    # Load dataset
    print("Loading jewelry dataset...")
    dataset = JewelryPromptDataset(
        csv_path,
        image_root=config.image_root,
        bracelets_only=config.bracelets_only,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    tokenizer = diffusion_model.pipeline.tokenizer
    text_encoder = diffusion_model.pipeline.text_encoder
    vae = diffusion_model.pipeline.vae
    unet = diffusion_model.pipeline.unet
    noise_scheduler = diffusion_model.pipeline.scheduler

    # --- 2. CAST MODELS TO BF16 ---
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    # Freeze non-LoRA modules.
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable LoRA params found on UNet.")

    optimizer = AdamW(trainable_params, lr=config.lr)

    history = []
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        unet.train()
        running_loss = 0.0
        num_steps = 0

        for batch in tqdm(dataloader, desc=f"Train epoch {epoch + 1}"):
            prompts = batch["prompt"]
            pixel_values = batch["pixel_values"].to(device)
            pixel_values = pixel_values.to(dtype=vae.dtype)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                input_ids = text_inputs.input_ids.to(device)
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            ).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()

            running_loss += float(loss.detach().item())
            num_steps += 1

        sample_prompts = dataset.df["prompt"].head(4).tolist()

        torch.cuda.empty_cache()
        
        print("Generating sample images...")
        sample_images = diffusion_model.generate_batch(
            sample_prompts, num_inference_steps=20, num_images_per_prompt=1
        )

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": (running_loss / max(1, num_steps)),
            "num_sample_images": len(sample_images),
        }
        history.append(epoch_metrics)

        print("DONE APPENDING")

        print(json.dumps(epoch_metrics, indent=2))

        # Save LoRA checkpoint directory
        ckpt_path = os.path.join(
            output_dir, "checkpoints", f"lora_epoch_{epoch+1}"
        )
        diffusion_model.save_lora(ckpt_path)
        print(f"Saved LoRA checkpoint to {ckpt_path}")

        # Save sample images
        for i, img in enumerate(sample_images):
            img_path = os.path.join(
                output_dir, f"sample_epoch{epoch+1}_img{i}.png"
            )
            img.save(img_path)

    # Save final config and history
    save_json(
        asdict(config),
        os.path.join(output_dir, "train_config.json"),
    )
    save_json(
        {"history": history},
        os.path.join(output_dir, "train_history.json"),
    )
    print(f"\nTraining complete. Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune diffusion LoRA")
    parser.add_argument("--csv", type=str, default="data/catalog_clip_ready.csv", help="Input CSV path")
    parser.add_argument("--image-root", type=str, default="data/images", help="Local image root (expects image_path from CSV relative to this directory)")
    parser.add_argument("--output", type=str, default="models/diffusion_lora", help="Output directory")
    parser.add_argument("--diffusion-model", type=str, default="runwayml/stable-diffusion-v1-5", help="HF diffusion model ID (default: SD v1.5 ~4GB; use 'stabilityai/stable-diffusion-xl-base-1.0' for SDXL ~13GB)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--bracelets-only", action="store_true", help="Train only on bracelet rows")
    parser.add_argument("--all-categories", action="store_true", help="Disable bracelet-only filter")
    args = parser.parse_args()

    bracelets_only = True
    if args.all_categories:
        bracelets_only = False
    elif args.bracelets_only:
        bracelets_only = True

    cfg = DiffusionTrainConfig(
        csv_path=args.csv,
        image_root=args.image_root,
        output_dir=args.output,
        diffusion_model_id=args.diffusion_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        bracelets_only=bracelets_only,
    )
    train_diffusion_lora(cfg)
