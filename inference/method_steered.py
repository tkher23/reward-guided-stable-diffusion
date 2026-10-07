"""Method 1: Our steered diffusion (SD1.5/SDXL + LoRA + oracle steering)"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.diffusion_steering import CostSteeredGenerator, SteeringConfig
from inference.oracle_inference import OracleInference
from models.model_diffusion import DiffusionGeneratorLoRA


class SteeredDiffusionMethod:
    """Generate images using gradient-based oracle steering."""

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
            print(f"Loading steered diffusion from HuggingFace: {hf_model_id}")
            self.diffusion = DiffusionGeneratorLoRA(
                model_id=hf_model_id,
                device=self.device,
            )
        else:
            print(f"Loading steered diffusion from local LoRA: {diffusion_lora_dir}")
            self.diffusion = DiffusionGeneratorLoRA(
                model_id=diffusion_model_id,
                device=self.device,
            )
            self.diffusion.setup_lora()
            self.diffusion.load_lora(str(Path(diffusion_lora_dir).resolve()))

        self.oracle = OracleInference(oracle_checkpoint, device=self.device)
        print(f"Loaded oracle from {oracle_checkpoint}")

        self.generator = CostSteeredGenerator(
            pipeline=self.diffusion.pipeline,
            oracle=self.oracle,
            device=self.device,
        )

    def generate(self, prompt: str, target_price: float, num_inference_steps: int = 30) -> tuple:
        """Returns: (PIL.Image, predicted_price)"""
        config = SteeringConfig(
            target_price=target_price,
            num_inference_steps=num_inference_steps,
        )
        image = self.generator.generate(prompt, config)
        predicted_price = float(self.oracle.score_pil_images([image]).item())
        return image, predicted_price
