from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import PIL.Image

from preference.inference.preference_oracle_inference import PreferenceOracleInference


@dataclass
class PreferenceSteeringConfig:
    """Configuration for preference-steered generation."""

    steering_strength: float = 1.0
    steer_every_k_steps: int = 1
    max_grad_norm: float = 1.0
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    image_size: int = 512  # output resolution; latent spatial size = image_size // 8
    debug_log: bool = False
    debug_log_every: int = 1


class PreferenceSteeredGenerator:
    """Inference-time preference steering using x0 estimates and preference oracle."""

    def __init__(
        self,
        pipeline,
        preference_oracle_inference: PreferenceOracleInference,
        device: str = "cuda",
    ) -> None:
        self.pipeline = pipeline
        self.oracle = preference_oracle_inference
        self.device = device

        if self.device == "cuda":
            self.pipeline.vae.to(self.device)
            self.pipeline.enable_vae_slicing()

    def _encode_prompt(self, prompt: str):
        """Encode text prompt, compatible with SD1.x and SDXL."""
        p = self.pipeline
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
        """Decode latents to image tensor."""
        p = self.pipeline
        scale = getattr(p.vae.config, "scaling_factor", 0.13025)
        image = p.vae.decode(latents / scale).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def _init_latents(self, image_size: int = 512) -> torch.Tensor:
        """Initialize random latents sized for the requested output resolution."""
        p = self.pipeline
        h = w = image_size // 8  # VAE downsamples by 8x (64 for 512px, 96 for 768px)
        return torch.randn(
            (1, p.unet.config.in_channels, h, w),
            device=self.device,
            dtype=p.unet.dtype,
        ) * p.scheduler.init_noise_sigma

    def generate(
        self,
        prompt: str,
        config: PreferenceSteeringConfig,
        user_preferences: Optional[torch.Tensor] = None,
        step_logger=None,
    ) -> PIL.Image.Image:
        """
        Generate image with preference steering.

        Args:
            prompt: text prompt for generation
            config: PreferenceSteeringConfig with steering parameters
            user_preferences: [num_features] per-user preference vector

        Returns:
            PIL Image
        """
        print(f"--- GENERATING WITH PREFERENCE STEERING ---")
        print(f"Prompt: {prompt}")
        if user_preferences is not None:
            print(f"User preferences: {user_preferences.tolist()}")

        p = self.pipeline
        p.scheduler.set_timesteps(config.num_inference_steps, device=self.device)

        # Encode prompt
        enc = self._encode_prompt(prompt)
        prompt_embeds = None
        negative_prompt_embeds = None
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None

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

        prompt_embeds = prompt_embeds.to(device=self.device, dtype=p.unet.dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=self.device, dtype=p.unet.dtype
            )
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(
                device=self.device, dtype=p.unet.dtype
            )
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                device=self.device, dtype=p.unet.dtype
            )

        latents = self._init_latents(getattr(config, "image_size", 512))

        # Main denoising loop
        for step_idx, t in enumerate(p.scheduler.timesteps):
            do_cfg = config.guidance_scale > 1.0

            # Determine if we steer at this step
            is_k_step = (
                config.steer_every_k_steps > 0
                and (step_idx % config.steer_every_k_steps == 0)
            )
            steer_now = is_k_step

            latents_in = latents.detach().requires_grad_(steer_now)
            latent_model_input = torch.cat([latents_in] * 2) if do_cfg else latents_in
            latent_model_input = p.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.to(dtype=p.unet.dtype)

            if do_cfg:
                if negative_prompt_embeds is None:
                    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                current_prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                )
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
                        negative_pooled_prompt_embeds = torch.zeros_like(
                            pooled_prompt_embeds
                        )
                    current_pooled_prompt_embeds = torch.cat(
                        [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
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
                noise_pred = noise_uncond + config.guidance_scale * (
                    noise_text - noise_uncond
                )

            step_out = p.scheduler.step(noise_pred, t, latents_in, return_dict=True)
            prev_latents = step_out.prev_sample

            # PREFERENCE STEERING
            if steer_now:
                # Estimate clean image
                x0_pred = step_out.pred_original_sample
                image_pred = self._decode_latents_to_image_tensor(x0_pred)

                # Compute preference score
                pref_score = self.oracle.predict_preference_score_tensor(
                    image_pred, user_preferences
                )

                if config.debug_log and (step_idx % max(config.debug_log_every, 1) == 0):
                    print(
                        f"[steer] Step {step_idx:03d} | Preference Score: {pref_score.item():.4f}"
                    )

                # Loss: maximize preference score
                loss = -pref_score

                # Gradient w.r.t. latent
                grad = torch.autograd.grad(
                    loss, latents_in, retain_graph=False, create_graph=False
                )[0]

                # Apply gradient update
                applied = False
                logged_grad_norm = None
                if torch.isfinite(grad).all():
                    grad_norm = grad.norm().clamp_min(1e-8)
                    grad = grad / grad_norm
                    grad = torch.clamp(
                        grad, -config.max_grad_norm, config.max_grad_norm
                    )

                    steering_step = (config.steering_strength * grad.detach()).to(
                        prev_latents.dtype
                    )
                    prev_latents = prev_latents - steering_step

                    # Safety clamp
                    prev_latents = torch.clamp(prev_latents, -4.0, 4.0)
                    applied = True
                    logged_grad_norm = float(grad_norm.item())

                    if config.debug_log:
                        print(
                            f"[stats] Grad norm: {logged_grad_norm:.4f} | Steering applied"
                        )
                else:
                    if config.debug_log:
                        print(
                            f"[steer] Warning: Non-finite grad at step {step_idx}. Skipping."
                        )

                # Per-step metric hook (records the steering trajectory of this image)
                if step_logger is not None:
                    step_logger(
                        step_idx=step_idx,
                        timestep=float(t),
                        pref_score=float(pref_score.item()),
                        grad_norm=logged_grad_norm,
                        steered=applied,
                    )

            latents = prev_latents.detach()

        # Decode final latent
        with torch.no_grad():
            scale = getattr(p.vae.config, "scaling_factor", 0.13025)
            image = p.vae.decode(latents / scale).sample
            pil = p.image_processor.postprocess(image, output_type="pil")[0]

        return pil
