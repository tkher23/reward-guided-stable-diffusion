"""Method 2: LoRA fine-tuned diffusion (no steering, no oracle guidance)"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_diffusion import DiffusionGeneratorLoRA
from inference.oracle_inference import OracleInference


class LoRATunedDiffusionNoSteering:
    """Generate images using LoRA fine-tuned model without oracle steering.

    Price is included in the prompt text but no gradient-based steering occurs.
    """

    def __init__(
        self,
        oracle_checkpoint: str,
        diffusion_source: str = "hf",
        hf_model_id: str = None,
        diffusion_lora_dir: str = None,
        diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
    ):
        self.device = device

        if diffusion_source == "hf" and hf_model_id:
            print(f"Loading LoRA diffusion from HuggingFace: {hf_model_id}")
            self.diffusion = DiffusionGeneratorLoRA(
                model_id=hf_model_id,
                device=self.device,
            )
        else:
            print(f"Loading LoRA diffusion from local: {diffusion_lora_dir}")
            self.diffusion = DiffusionGeneratorLoRA(
                model_id=diffusion_model_id,
                device=self.device,
            )
            self.diffusion.setup_lora()
            self.diffusion.load_lora(str(Path(diffusion_lora_dir).resolve()))

        self.oracle = OracleInference(oracle_checkpoint, device=self.device)
        print(f"Loaded oracle from {oracle_checkpoint}")

    def generate(self, prompt: str, target_price: float, num_inference_steps: int = 30) -> tuple:
        """Returns: (PIL.Image, predicted_price)"""
        augmented_prompt = f"{prompt}, ${target_price:.0f}"
        images = self.diffusion.generate_batch(
            [augmented_prompt],
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
        )
        image = images[0]
        predicted_price = float(self.oracle.score_pil_images([image]).item())
        return image, predicted_price
