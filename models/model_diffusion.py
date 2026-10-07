from __future__ import annotations

import os
from pathlib import Path

# Force transformers/diffusers to avoid TensorFlow backend in this environment.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from peft import get_peft_model, LoraConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "diffusion_base"


def _is_local_model_path(model_id_or_path: str) -> bool:
    return Path(model_id_or_path).exists()


def _safe_cache_dir_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "__")


def _resolve_base_model_source(model_id_or_path: str) -> str:
    if _is_local_model_path(model_id_or_path):
        return str(Path(model_id_or_path).resolve())
    return model_id_or_path


class DiffusionGeneratorLoRA(torch.nn.Module):
    """SDXL wrapper with LoRA fine-tuning support."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_rank: int = 64,
        lora_alpha: int = 128,
        device: str = "cuda",
        cpu_fallback_model_id: str = "runwayml/stable-diffusion-v1-5",
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        model_id = _resolve_base_model_source(model_id)

        if device == "cpu" and any(x in model_id.lower() for x in ["sdxl", "stable-diffusion-xl"]):
            print(
                f"CPU mode detected. Falling back from {model_id} to {cpu_fallback_model_id} "
                "to reduce memory usage."
            )
            cpu_fallback_model_id = _resolve_base_model_source(cpu_fallback_model_id)
            model_id = cpu_fallback_model_id
        self.model_id = model_id

        # Load base pipeline with local cache directory
        cache_dir = str(LOCAL_MODEL_CACHE_DIR)
        print(f"Using HF cache directory: {cache_dir}")
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            safety_checker=None,
        )
        self.pipeline = self.pipeline.to(device)

        # Hard-disable the safety path so generated samples are not replaced with black images.
        if hasattr(self.pipeline, "safety_checker"):
            self.pipeline.safety_checker = None
        if hasattr(self.pipeline, "feature_extractor"):
            self.pipeline.feature_extractor = None
        if hasattr(self.pipeline, "requires_safety_checker"):
            self.pipeline.requires_safety_checker = False

        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()
        if hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()

        # Optional: Use faster scheduler for inference
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )

    def forward(self, prompt: str, num_inference_steps: int = 30, guidance_scale: float = 7.5):
        """Generate image from text prompt."""
        with torch.no_grad():
            image = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=768,
                width=768,
            ).images[0]
        return image

    def generate_batch(
        self,
        prompts: list[str],
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        **kwargs  # <--- Add **kwargs here to catch the callback
    ) -> list:
        """Generate multiple images from prompts."""
        images = []
        with torch.no_grad():
            for prompt in prompts:
                for _ in range(num_images_per_prompt):
                    image = self.pipeline(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=768,
                        width=768,
                        **kwargs
                    ).images[0]
                    images.append(image)
        return images

    def setup_lora(self, r: int = None, lora_alpha: int = None) -> None:
        """Apply LoRA to UNet for efficient fine-tuning."""
        r = r or self.lora_rank
        lora_alpha = lora_alpha or self.lora_alpha

        unet = self.pipeline.unet
        lora_kwargs = dict(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.05,
            bias="none",
        )
        lora_config = LoraConfig(**lora_kwargs)

        # Prefer diffusers native adapter path to avoid PEFT wrapper forward-signature
        # mismatches on UNet (e.g. unexpected 'input_ids').
        if hasattr(unet, "add_adapter"):
            unet.add_adapter(lora_config)
            self.pipeline.unet = unet
        else:
            self.pipeline.unet = get_peft_model(unet, lora_config)

        if hasattr(self.pipeline.unet, "print_trainable_parameters"):
            self.pipeline.unet.print_trainable_parameters()

    def save_lora(self, save_dir: str) -> None:
        """Save LoRA weights only."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # Prefer diffusers-native LoRA serialization when available.
        if hasattr(self.pipeline, "save_lora_weights"):
            self.pipeline.save_lora_weights(save_dir)
        else:
            # Fallback for older setups that rely on PEFT-wrapped UNet.
            self.pipeline.unet.save_pretrained(save_dir)

    def load_lora(self, lora_dir: str) -> None:
        """Load LoRA weights into UNet."""
        if hasattr(self.pipeline, "load_lora_weights"):
            self.pipeline.load_lora_weights(lora_dir)
            return

        from peft import PeftModel

        self.pipeline.unet = PeftModel.from_pretrained(
            self.pipeline.unet, lora_dir
        )
