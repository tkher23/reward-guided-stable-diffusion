from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PreferenceExperimentConfig:
    """Configuration for the preference steering experiment."""

    # Data
    num_zappos_images: int = 100
    feature_concepts: list[str] = None

    # Feature space: how images are represented for the preference model.
    #   "concepts"  -> low-dim, interpretable: per-attribute CLIP scores
    #                  (colorful/chunky/clean), z-scored across the dataset.
    #   "embedding" -> full L2-normalized CLIP image embedding (512-d).
    # If compare_feature_spaces is True, the experiment runs once per entry in
    # feature_spaces and reports them side by side; otherwise it uses
    # feature_space alone.
    feature_space: str = "concepts"
    compare_feature_spaces: bool = False
    feature_spaces: list[str] = field(
        default_factory=lambda: ["concepts", "embedding"]
    )

    # Training phase (learning theta)
    num_synthetic_train_users: int = 50
    interactions_per_train_user: int = 100
    theta_learning_rate: float = 0.01

    # Test phase (held-out users)
    num_synthetic_test_users: int = 3
    rounds_per_test_user: int = 50
    user_learning_rate: float = 0.05

    # Restrict evaluation to a single configuration instead of the default
    # fixed + adaptive comparison. One of: "nosteer" (base-model baseline, no
    # preference steering), "fixed", "adaptive". None = run fixed + adaptive.
    only_mode: str | None = None

    # Preference model
    adapt_theta: bool = False  # Toggle: fixed theta vs. adaptive theta

    # Steering
    steering_strength: float = 1.0  # 1.0 = full (100%) preference steering
    steer_every_k_steps: int = 1
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    max_grad_norm: float = 1.0

    # Generation: when True, the test phase actually runs the diffusion model
    # with preference steering and saves a real image every round. When False,
    # it uses the lightweight feature-space proxy (no images).
    use_real_diffusion: bool = True
    generation_prompt: str = "a sneaker"
    debug_log_steering: bool = False  # log per-step steering score/grad norm

    # Paths
    sneaker_data_dir: str = "sneakerdata"
    preference_models_dir: str = "preference/models_checkpoints"
    results_dir: str = "preference/results"

    # Diffusion model
    diffusion_model_id: str = "runwayml/stable-diffusion-v1-5"
    image_size: int = 512  # generation resolution; latent = image_size // 8
    lora_checkpoint: str = None  # Path to trained sneaker LoRA weights dir
    # Deliberately steer the vanilla base model (no LoRA). When True the LoRA is
    # never loaded even if lora_checkpoint is set -- inference runs on plain SD1.5.
    base_only: bool = False

    # Artifact export / reuse (LoRA + preference model bundle)
    export_dir: str = None       # write a reusable bundle here after the run
    hf_sync_target: str = None   # e.g. hf://buckets/tkher/preferencemodels
    load_bundle: str = None      # load LoRA + theta from a bundle; skip theta training

    # Metrics: record every example + steering step to results/metrics.sqlite
    log_metrics_db: bool = True

    def __post_init__(self):
        if self.feature_concepts is None:
            self.feature_concepts = [
                "colorful bright sneaker",
                "chunky platform sneaker",
                "clean minimal sneaker",
            ]
