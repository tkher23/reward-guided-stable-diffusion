from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import PIL.Image

from inference.oracle_inference import OracleInference


@dataclass
class SteeringConfig:
    target_price: Optional[float] = None
    steering_strength: float = 1
    steer_every_k_steps: int = 1
    max_grad_norm: float = 1.0
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    debug_log: bool = False
    debug_log_every: int = 1


class CostSteeredGenerator:
    """Inference-time cost steering using x0 estimates and a frozen oracle."""

    def __init__(self, pipeline, oracle: OracleInference, device: str = "cuda") -> None:
        self.pipeline = pipeline
        self.oracle = oracle
        self.device = device

        if hasattr(oracle, 'model'):
            oracle.model.to(self.device)
        # FOR MEMORY STABILITY
        if self.device == "cuda":
            self.pipeline.vae.to(self.device)
            # self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_vae_slicing()

    def _encode_prompt(self, prompt: str):
        p = self.pipeline
        # Handles SDXL and SD1.x signature variants.
        try:
            enc = p.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="",
            )
        except TypeError:
            enc = p.encode_prompt(
                prompt,
                self.device,
                1,
                True,
                "",
            )
        return enc

    def _decode_latents_to_image_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        p = self.pipeline
        # SDXL uses 0.13025. SD1.5 uses 0.18215. 
        # This getattr will handle it, but ensure your config actually has it.
        scale = getattr(p.vae.config, "scaling_factor", 0.13025) 
        image = p.vae.decode(latents / scale).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def _init_latents(self) -> torch.Tensor:
        p = self.pipeline
        # FORCE 96x96 to match the 768x768 resolution used in your generate_batch
        h = 96
        w = 96
        
        # --- COMMENT OUT OR REMOVE THIS BLOCK ---
        # sample_size = getattr(p.unet.config, "sample_size", None)
        # if isinstance(sample_size, int):
        #     h = sample_size
        #     w = sample_size
        # ----------------------------------------

        return torch.randn(
            (1, p.unet.config.in_channels, h, w),
            device=self.device,
            dtype=p.unet.dtype,
        ) * p.scheduler.init_noise_sigma

    def generate(self, prompt: str, config: SteeringConfig):
        print(f"--- STARTING GENERATION FOR: {prompt} ---", flush=True)
        p = self.pipeline
        p.scheduler.set_timesteps(config.num_inference_steps, device=self.device)

        # Extract prompt embeddings from SDXL/SD1.x compatible encode_prompt return.
        prompt_embeds = None
        negative_prompt_embeds = None
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None

        enc = self._encode_prompt(prompt)
        if isinstance(enc, tuple):
            if len(enc) >= 1:
                prompt_embeds = enc[0]
            if len(enc) >= 2:
                negative_prompt_embeds = enc[1]
            if len(enc) >= 3:
                pooled_prompt_embeds = enc[2]
            if len(enc) >= 4:
                negative_pooled_prompt_embeds = enc[3]
        else:
            prompt_embeds = enc

        if prompt_embeds is None:
            raise RuntimeError("Failed to build prompt embeddings for steering loop.")

        prompt_embeds = prompt_embeds.to(device=self.device, dtype=p.unet.dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=self.device, dtype=p.unet.dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=self.device, dtype=p.unet.dtype)
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=self.device, dtype=p.unet.dtype)

        latents = self._init_latents()
        if config.target_price is None:
            raise ValueError("SteeringConfig.target_price must be set before calling generate().")
        target_log = torch.log(torch.tensor(float(config.target_price), device=self.device))

        # Steer from the very first step with constant strength.
        warmup_steps = 0

        # Precompute 5 evenly-spaced step indices for x0 preview saving.
        n = config.num_inference_steps
        preview_steps = {int(i * (n - 1) / 4) for i in range(5)}
        os.makedirs("inference/pred_images", exist_ok=True)

        for step_idx, t in enumerate(p.scheduler.timesteps):
            do_cfg = config.guidance_scale > 1.0
            
            # --- UPDATED STEERING LOGIC ---
            is_past_warmup = step_idx >= warmup_steps
            is_k_step = config.steer_every_k_steps > 0 and (step_idx % config.steer_every_k_steps == 0)
            steer_now = is_past_warmup and is_k_step

            latents_in = latents.detach().requires_grad_(steer_now)
            latent_model_input = torch.cat([latents_in] * 2) if do_cfg else latents_in
            latent_model_input = p.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.to(dtype=p.unet.dtype)

            if do_cfg:
                if negative_prompt_embeds is None:
                    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                current_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            else:
                current_prompt_embeds = prompt_embeds

            unet_kwargs = {
                "sample": latent_model_input,
                "timestep": t,
                "encoder_hidden_states": current_prompt_embeds,
            }

            if pooled_prompt_embeds is not None:
                if do_cfg:
                    if negative_pooled_prompt_embeds is None:
                        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
                    current_pooled_prompt_embeds = torch.cat(
                        [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                        dim=0,
                    )
                else:
                    current_pooled_prompt_embeds = pooled_prompt_embeds

                time_ids = torch.tensor(
                    [[768, 768, 0, 0, 768, 768]],
                    device=self.device,
                    dtype=current_prompt_embeds.dtype,
                )
                time_ids = time_ids.repeat(current_prompt_embeds.shape[0], 1)
                unet_kwargs["added_cond_kwargs"] = {
                    "text_embeds": current_pooled_prompt_embeds,
                    "time_ids": time_ids,
                }

            noise_pred = p.unet(**unet_kwargs).sample
            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + config.guidance_scale * (noise_text - noise_uncond)

            step_out = p.scheduler.step(noise_pred, t, latents_in, return_dict=True)
            prev_latents = step_out.prev_sample

            # Save decoded x0 estimate at 5 evenly-spaced steps for inspection.
            if step_idx in preview_steps:
                with torch.no_grad():
                    x0_preview = self._decode_latents_to_image_tensor(step_out.pred_original_sample)
                    arr = (x0_preview.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
                    PIL.Image.fromarray(arr).save(f"inference/pred_images/step_{step_idx:03d}.png")
                    print(f"[preview] Saved x0 preview at step {step_idx}", flush=True)

            if steer_now:
                current_strength = config.steering_strength
                # Estimate x0 and decode to image
                x0_pred = step_out.pred_original_sample
                image_pred = self._decode_latents_to_image_tensor(x0_pred)
                
                # 2. Predict cost
                pred_log = self.oracle.predict_log_price_tensor(image_pred)
                print(f"[steer] Step {step_idx:03d} | Predicted Log Price: {pred_log.item():.4f} | Target Log Price: {target_log.item():.4f} | Current Strength: {current_strength:.4f}")
                loss = (pred_log.float() - target_log.float()).pow(2).mean()

                # 3. Debug logging
                if config.debug_log and (step_idx % max(config.debug_log_every, 1) == 0):
                    pred_price = float(torch.exp(pred_log.detach()).mean().item())
                    target_price = float(config.target_price)
                    print(
                        f"[steer] step={step_idx:03d} t={int(t)} "
                        f"pred_price={pred_price:.2f} target={target_price:.2f} "
                        f"loss={float(loss.detach().item()):.5f}"
                    )

                # 4. Calculate Gradient
                grad = torch.autograd.grad(loss, latents_in, retain_graph=False, create_graph=False)[0]
                
                # 5. Apply the Nudge with Statistical Renormalization
                if torch.isfinite(grad).all():
                    # Normalize and clamp the gradient
                    grad_norm = grad.norm().clamp_min(1e-8)
                    grad = grad / grad_norm
                    grad = torch.clamp(grad, -config.max_grad_norm, config.max_grad_norm)

                    # Apply the nudge
                    steering_step = (current_strength * grad.detach()).to(prev_latents.dtype)
                    prev_latents = prev_latents - steering_step

                    with torch.no_grad():
                        l_mean = prev_latents.mean().item()
                        l_std = prev_latents.std().item()
                        g_mean = grad.mean().item()
                        g_std = grad.std().item()
                        print(f"[stats] Latent Mean: {l_mean:.4f}, Std: {l_std:.4f} | Grad Mean: {g_mean:.4f}, Std: {g_std:.4f}")

                    # Safety clamp to keep values within SDXL's expected range
                    prev_latents = torch.clamp(prev_latents, -4.0, 4.0)
                else:
                    if config.debug_log:
                        print(f"[steer] Warning: Non-finite grad detected at step {step_idx}. Skipping steering.")

            latents = prev_latents.detach()
    
        with torch.no_grad():
            scale = getattr(p.vae.config, "scaling_factor", 0.13025)
            image = p.vae.decode(latents / scale).sample  # [-1, 1]; let postprocess denormalize
            pil = p.image_processor.postprocess(image, output_type="pil")[0]
        return pil