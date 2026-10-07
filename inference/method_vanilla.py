"""Method 3: Vanilla SD1.5/SDXL (no LoRA, no steering)"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_diffusion import DiffusionGeneratorLoRA
from inference.oracle_inference import OracleInference


class VanillaDiffusion:
    """Generate images using vanilla SD1.5/SDXL without any fine-tuning or steering.

    The prompt includes the target price information, but model is unmodified base.
    """

    def __init__(
        self,
        oracle_checkpoint: str = "models/oracle/checkpoints/best.pt",
        diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
    ):
        self.device = device

        # Load base diffusion WITHOUT LoRA
        self.diffusion = DiffusionGeneratorLoRA(
            model_id=diffusion_model_id,
            device=self.device,
        )
        # NOTE: we do NOT call setup_lora() or load_lora() — use base model as-is
        print(f"Loaded vanilla diffusion model: {diffusion_model_id}")

        # Load oracle for evaluation
        self.oracle = OracleInference(oracle_checkpoint, device=self.device)
        print(f"Loaded oracle from {oracle_checkpoint}")

    def generate(self, prompt: str, target_price: float, num_inference_steps: int = 30) -> tuple:
        """Generate image using vanilla base model with price included in prompt.

        Args:
            prompt: base description
            target_price: target price (included in prompt text)
            num_inference_steps: diffusion steps

        Returns: (PIL.Image, predicted_price)
        """
        # Augment prompt to include price
        augmented_prompt = f"{prompt}, ${target_price:.0f}"

        # Generate without steering, using base model
        images = self.diffusion.generate_batch(
            [augmented_prompt],
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
        )
        image = images[0]
        predicted_price = float(self.oracle.score_pil_images([image]).item())
        return image, predicted_price
