# Stage 1 of the preference pipeline: fine-tune an SD1.5 LoRA on the 100
# Zappos sneaker images so generation produces on-distribution shoes.
#
#   python preference/training/train_sneaker_lora.py \
#       --image-dir sneakerdata/images/shoes_100 \
#       --output preference/models_checkpoints/sneaker_lora
#
# Then point the experiment at it:
#   python preference/experiments/full_preference_experiment.py \
#       --smoke --lora preference/models_checkpoints/sneaker_lora

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Avoid the TensorFlow backend in transformers/diffusers.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_diffusion import DiffusionGeneratorLoRA


class SneakerImageDataset(Dataset):
    """Yields (constant prompt, pixel_values) for every .jpg in a folder."""

    def __init__(
        self,
        image_dir: str,
        prompt: str = "a sneaker",
        image_size: int = 512,
        max_images: int | None = None,
    ):
        self.paths = sorted(Path(image_dir).glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"No .jpg images found in {image_dir}")
        if max_images is not None:
            self.paths = self.paths[:max_images]
        self.prompt = prompt
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.paths[idx]).convert("RGB")
        return {"prompt": self.prompt, "pixel_values": self.transform(image)}


def train_sneaker_lora(
    image_dir: str = "sneakerdata/images/shoes_100",
    output_dir: str = "preference/models_checkpoints/sneaker_lora",
    model_id: str = "runwayml/stable-diffusion-v1-5",
    prompt: str = "a sneaker",
    image_size: int = 512,
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    seed: int = 42,
    max_images: int | None = None,
    num_samples: int = 4,
) -> str:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Using device: {device}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_id} with LoRA...")
    diffusion = DiffusionGeneratorLoRA(
        model_id=model_id,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=device,
    )
    diffusion.setup_lora()

    dataset = SneakerImageDataset(
        image_dir, prompt=prompt, image_size=image_size, max_images=max_images
    )
    print(f"Training LoRA on {len(dataset)} images with prompt {prompt!r}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    pipe = diffusion.pipeline
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    scheduler = pipe.scheduler

    text_encoder.to(device, dtype=cast_dtype)
    vae.to(device, dtype=cast_dtype)
    unet.to(device, dtype=cast_dtype)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    trainable = [p for p in unet.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA params found on the UNet.")
    optimizer = AdamW(trainable, lr=lr)

    for epoch in range(epochs):
        unet.train()
        running, steps = 0.0, 0
        for batch in tqdm(dataloader, desc=f"epoch {epoch + 1}/{epochs}"):
            pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                tok = tokenizer(
                    batch["prompt"],
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                encoder_hidden_states = text_encoder(tok.input_ids.to(device))[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy = scheduler.add_noise(latents, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            pred = unet(noisy, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(pred.float(), noise.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            running += float(loss.detach().item())
            steps += 1

        print(f"epoch {epoch + 1}: train_loss={running / max(1, steps):.4f}")

    # Save LoRA in diffusers format. diffusers 0.30 requires the LoRA layers to
    # be passed explicitly -- pipe.save_lora_weights(dir) with no layers raises
    # "You must pass at least one of unet_lora_layers ...". We extract them from
    # the adapter exactly like the official diffusers SD1.5 LoRA example, which
    # writes pytorch_lora_weights.safetensors that load_lora_weights reads back.
    from peft.utils import get_peft_model_state_dict
    from diffusers.utils import convert_state_dict_to_diffusers

    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unet)
    )
    pipe.save_lora_weights(
        save_directory=str(out),
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )
    print(f"\nSaved sneaker LoRA to {out}")

    # Dump a few sample generations so LoRA quality can be eyeballed before
    # committing to stages 2-3. Calls the pipeline directly at image_size
    # (generate_batch hardcodes 768px, which is wrong for SD1.5 @ 512).
    if num_samples > 0:
        samples_dir = out / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        unet.eval()
        print(f"Generating {num_samples} sample image(s) at {image_size}px...")
        with torch.no_grad():
            for i in range(num_samples):
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=image_size,
                    width=image_size,
                ).images[0]
                image.save(samples_dir / f"sample_{i}.png")
        print(f"Saved {num_samples} sample(s) to {samples_dir}")

    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an SD1.5 LoRA on the 100 sneaker images (stage 1)."
    )
    parser.add_argument("--image-dir", default="sneakerdata/images/shoes_100")
    parser.add_argument("--output", default="preference/models_checkpoints/sneaker_lora")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt", default="a sneaker")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Use only the first N images (quick sanity check)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Sample images to generate after training (0 to skip)",
    )
    args = parser.parse_args()

    train_sneaker_lora(
        image_dir=args.image_dir,
        output_dir=args.output,
        model_id=args.model_id,
        prompt=args.prompt,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_images=args.max_train_images,
        num_samples=args.num_samples,
    )
