# python inference/generate_designs.py

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_diffusion import DiffusionGeneratorLoRA
from evals.evaluate_diffusion import evaluate_generation_quality
from inference.oracle_inference import OracleInference
from inference.diffusion_steering import CostSteeredGenerator, SteeringConfig


class DesignInference:
    """Interactive interface for generating and ranking jewelry designs."""

    def __init__(
        self,
        diffusion_lora_dir: str = "models/diffusion_lora",
        clip_checkpoint: str = "models/clip_baseline/checkpoints/best.pt",
        oracle_checkpoint: Optional[str] = "models/oracle/checkpoints/best.pt",
        diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
        hf_diffusion_model_id: Optional[str] = None,
        hf_oracle_model_id: Optional[str] = None,
        oracle_source: str = "local",
        hf_token: Optional[str] = None,
        diffusion_source: str = "local-finetune",
        train_local_diffusion_first: bool = False,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        use_clip_eval: bool = False,
        enable_steering: bool = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_clip_eval = use_clip_eval
        self.enable_steering = enable_steering
        print(f"Using device: {self.device}")

        self.clip_model = None
        self.clip_processor = None
        if self.use_clip_eval:
            # CLIP is optional and only used for legacy/baseline evaluation.
            from transformers import CLIPProcessor
            from models.model_clip import CLIPFinetune

            print("Loading CLIP model...")
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model = CLIPFinetune(model_name=clip_model_name).to(self.device)
            if os.path.exists(clip_checkpoint):
                checkpoint = torch.load(clip_checkpoint, map_location=self.device)
                self.clip_model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded CLIP from {clip_checkpoint}")
            self.clip_model.eval()
        else:
            print("CLIP evaluation disabled (cost-focused mode).")

        # Load diffusion according to chosen source.
        if diffusion_source == "hf" and hf_diffusion_model_id:
            print(f"Loading diffusion directly from Hugging Face model: {hf_diffusion_model_id}")
            self.diffusion = DiffusionGeneratorLoRA(
                model_id=hf_diffusion_model_id,
                device=self.device,
            )
        else:
            if train_local_diffusion_first:
                print("Training local diffusion LoRA before inference (same pipeline as current training script)...")
                from training.train_diffusion_lora import DiffusionTrainConfig, train_diffusion_lora

                train_cfg = DiffusionTrainConfig(
                    output_dir=diffusion_lora_dir,
                    bracelets_only=True,
                )
                train_diffusion_lora(train_cfg)

            print(f"Loading base diffusion model ({diffusion_model_id}) + local LoRA...")
            self.diffusion = DiffusionGeneratorLoRA(
                model_id=diffusion_model_id,
                device=self.device,
            )
            self.diffusion.setup_lora()

            if os.path.exists(diffusion_lora_dir):
                try:
                    self.diffusion.load_lora(diffusion_lora_dir)
                    print(f"Loaded LoRA from {diffusion_lora_dir}")
                except Exception as e:
                    print(f"Could not load LoRA: {e}. Using fresh LoRA.")

        self.oracle = None
        self.steered_generator = None
        oracle_ckpt_path = oracle_checkpoint
        if oracle_source == "hf":
            oracle_repo = hf_oracle_model_id or hf_diffusion_model_id
            if not oracle_repo:
                raise ValueError(
                    "Oracle source is hf, but no HF repo id was provided. Set --hf-oracle-model-id "
                    "or use --hf-diffusion-model-id and --diffusion-source hf."
                )
            print(f"Downloading oracle checkpoint from Hugging Face repo: {oracle_repo}")
            oracle_ckpt_path = hf_hub_download(
                repo_id=oracle_repo,
                filename="oracle/best.pt",
                token=hf_token,
            )

        if oracle_ckpt_path and os.path.exists(oracle_ckpt_path):
            print("Loading Cost Oracle...")
            self.oracle = OracleInference(oracle_ckpt_path, device=self.device)
            if self.enable_steering:
                self.steered_generator = CostSteeredGenerator(
                    pipeline=self.diffusion.pipeline,
                    oracle=self.oracle,
                    device=self.device,
                )
        elif self.enable_steering:
            print("Warning: steering requested but oracle checkpoint not found. Using standard generation.")

    def generate_and_rank(
        self,
        prompt: str,
        num_images: int = 4,
        steer_cfg: Optional[SteeringConfig] = None,
    ) -> list[dict]:
        """Generate images and rank by cost alignment (or CLIP if enabled)."""

        cfg = steer_cfg or SteeringConfig()
        print(f"\n> Generating {num_images} designs for: '{prompt}'")
        if self.enable_steering and self.steered_generator and cfg.target_price is not None:
            images = []
            for _ in range(num_images):
                images.append(self.steered_generator.generate(prompt, cfg))
        else:
            images = self.diffusion.generate_batch(
                [prompt],
                num_images_per_prompt=num_images,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                callback_on_step_end=self.diagnostic_callback,
                callback_on_step_end_tensor_inputs=["latents"]
            )

        results = []
        for i, img in enumerate(images):
            entry = {
                "index": i + 1,
                "image": img,
            }

            if self.use_clip_eval and self.clip_model and self.clip_processor:
                metrics = evaluate_generation_quality(
                    self.clip_model,
                    [img],
                    [prompt],
                    self.clip_processor,
                    self.device,
                )
                entry.update(
                    {
                        "mean_similarity": metrics["mean_similarity"],
                        "binding_score": metrics["binding_score"],
                    }
                )

            if self.oracle is not None:
                price_pred = float(self.oracle.score_pil_images([img]).item())
                entry["predicted_price"] = price_pred
                if cfg.target_price is not None:
                    signed_error = price_pred - float(cfg.target_price)
                    entry["signed_error"] = signed_error
                    entry["price_error"] = abs(signed_error)

            results.append(entry)

        # Prefer cost-based ranking when target budget is available.
        if cfg.target_price is not None and self.oracle is not None:
            results = sorted(results, key=lambda x: x.get("price_error", float("inf")))
        elif self.use_clip_eval:
            results = sorted(results, key=lambda x: x.get("mean_similarity", -1.0), reverse=True)

        return results

    def save_results(
        self,
        results: list[dict],
        output_dir: str = "inference/outputs",
        prompt_name: str = "design",
    ) -> None:
        """Save generated images and metrics to disk."""

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for rank, result in enumerate(results, 1):
            score_for_name = result.get("price_error", result.get("mean_similarity", 0.0))
            img_path = os.path.join(
                output_dir, f"{prompt_name}_rank{rank}_score{score_for_name:.3f}.png"
            )
            result["image"].save(img_path)

            if "price_error" in result:
                signed_error = float(result.get("signed_error", 0.0))
                direction = "over" if signed_error >= 0 else "under"
                print(
                    f"  Rank {rank}: price={result.get('predicted_price', 0.0):.2f}, "
                    f"signed_error={signed_error:+.2f} ({direction}), "
                    f"abs_error={result['price_error']:.2f}"
                )
            elif "mean_similarity" in result:
                print(
                    f"  Rank {rank}: similarity={result['mean_similarity']:.3f}, "
                    f"binding={result.get('binding_score', 0.0):.3f}"
                )

    def interactive_session(self) -> None:
        """Start interactive CLI session."""

        print("\n" + "=" * 60)
        print("JEWELRY DESIGN GENERATOR")
        print("=" * 60)
        print("\nDescribe a jewelry design you'd like to generate.")
        print("Example: 'silver bracelet with blue stone accents and heart charm'")
        print("Commands: 'quit' to exit, 'help' for guidance\n")

        design_counter = 0
        while True:
            prompt = input("-> Design description: ").strip()

            if prompt.lower() == "quit":
                print("Goodbye!")
                break

            if prompt.lower() == "help":
                print(
                    "\nTips for best results:"
                    "\n- Include material (silver, gold, rose gold)"
                    "\n- Mention color (blue, green, red, etc)"
                    "\n- Describe shape/style (bracelet, necklace, ring, pendant)"
                    "\n- Add details (stones, pearls, engravings, charms)\n"
                )
                continue

            if not prompt:
                continue

            try:
                design_counter += 1
                steer_cfg = SteeringConfig()
                if self.enable_steering and self.oracle is not None:
                    budget_raw = input("-> Target budget (USD, blank for none): ").strip()
                    if budget_raw:
                        steer_cfg.target_price = float(budget_raw)

                results = self.generate_and_rank(
                    prompt,
                    num_images=4,
                    steer_cfg=steer_cfg,
                )

                print("\n> Generated designs (ranked by quality):\n")
                self.save_results(results, prompt_name=f"design_{design_counter}")

                for result in results:
                    if "price_error" in result:
                        signed_error = float(result.get("signed_error", 0.0))
                        direction = "over" if signed_error >= 0 else "under"
                        print(
                            f"  #{result['index']} -> "
                            f"Predicted Price: {result.get('predicted_price', 0.0):.2f} | "
                            f"Signed Error: {signed_error:+.2f} ({direction}) | "
                            f"Abs Error: {result['price_error']:.2f}"
                        )
                    elif "mean_similarity" in result:
                        print(
                            f"  #{result['index']} -> "
                            f"Alignment: {result['mean_similarity']:.3f} | "
                            f"Consistency: {result.get('binding_score', 0.0):.3f}"
                        )

            except Exception as e:
                print(f"Error generating designs: {e}")

    def batch_generate(
        self,
        prompts: list[str],
        output_dir: str = "inference/outputs",
    ) -> None:
        """Generate designs for multiple prompts in batch."""

        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}]")
            results = self.generate_and_rank(prompt, num_images=3)
            self.save_results(results, output_dir=output_dir, prompt_name=f"batch_{i}")

    # Ensure this is inside the class and has 'self'
    def diagnostic_callback(self, pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        with torch.no_grad():
            l_mean = latents.mean().item()
            l_std = latents.std().item()
            # Added flush=True to force the remote terminal to show it
            print(f"[BASE-STEP-{step_index:02d}] Mean: {l_mean:.4f}, Std: {l_std:.4f}", flush=True)
        return callback_kwargs
        
def main():
    """CLI entry point."""

    import argparse

    parser = argparse.ArgumentParser(description="Generate jewelry designs with diffusion + LoRA")
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch"],
        default="interactive",
        help="Run interactive session or batch generation",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Design prompts for batch mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference/outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default="models/diffusion_lora",
        help="Directory containing trained LoRA weights",
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base diffusion model id for local-finetune mode (default: SD v1.5)",
    )
    parser.add_argument(
        "--diffusion-source",
        choices=["hf", "local-finetune"],
        default="local-finetune",
        help="Use diffusion model from Hugging Face directly or local fine-tune path",
    )
    parser.add_argument(
        "--hf-diffusion-model-id",
        type=str,
        default="",
        help="HF model id to use when --diffusion-source hf",
    )
    parser.add_argument(
        "--oracle-source",
        choices=["local", "hf"],
        default="local",
        help="Load oracle checkpoint from local path or Hugging Face repo",
    )
    parser.add_argument(
        "--hf-oracle-model-id",
        type=str,
        default="",
        help="HF model id that contains oracle/best.pt (defaults to --hf-diffusion-model-id when omitted)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Optional HF token for private repos",
    )
    parser.add_argument(
        "--train-local-diffusion-first",
        action="store_true",
        help="If using local-finetune source, run local diffusion fine-tuning before inference",
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=str,
        default="models/clip_baseline/checkpoints/best.pt",
        help="Path to trained CLIP checkpoint",
    )
    parser.add_argument(
        "--oracle-checkpoint",
        type=str,
        default="models/oracle/checkpoints/best.pt",
        help="Path to trained price oracle checkpoint",
    )
    parser.add_argument(
        "--use-clip-eval",
        action="store_true",
        help="Enable CLIP-based evaluation metrics",
    )
    parser.add_argument(
        "--steering",
        action="store_true",
        help="Enable cost steering with the oracle",
    )
    parser.add_argument(
        "--target-price",
        type=float,
        default=None,
        help="Target price for steering/ranking",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=4,
        help="Number of images to generate per prompt",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Number of diffusion inference steps",
    )

    args = parser.parse_args()

    # Initialize inference engine
    engine = DesignInference(
        diffusion_lora_dir=args.lora_dir,
        clip_checkpoint=args.clip_checkpoint,
        oracle_checkpoint=args.oracle_checkpoint,
        diffusion_model_id=args.diffusion_model,
        diffusion_source=args.diffusion_source,
        hf_diffusion_model_id=args.hf_diffusion_model_id or None,
        hf_oracle_model_id=args.hf_oracle_model_id or None,
        oracle_source=args.oracle_source,
        hf_token=args.hf_token or None,
        train_local_diffusion_first=args.train_local_diffusion_first,
        use_clip_eval=args.use_clip_eval,
        enable_steering=args.steering,
    )

    if args.mode == "interactive":
        engine.interactive_session()
    elif args.mode == "batch":
        if not args.prompts:
            print("Error: --prompts required for batch mode")
            sys.exit(1)
        steer_cfg = SteeringConfig(
            target_price=args.target_price,
            num_inference_steps=args.steps,
        )
        for p in args.prompts:
            results = engine.generate_and_rank(
                prompt=p,
                num_images=args.num_images,
                steer_cfg=steer_cfg,
            )
            engine.save_results(results, output_dir=args.output, prompt_name="batch")


if __name__ == "__main__":
    main()
