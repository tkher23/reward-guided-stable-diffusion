from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

# Add repo root to path so imports work from anywhere
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from preference.experiments.config import PreferenceExperimentConfig
from preference.data_prep.download_zappos import download_zappos_dataset
from preference.data_prep.label_features_with_ai import label_features_with_clip
from preference.models.feature_extractor import (
    DifferentiableFeatureExtractor,
    validate_feature_space,
)
from preference.simulation.simulate_training_phase import simulate_training_phase
from preference.simulation.simulate_test_phase import simulate_test_phase
from preference.evals.evaluate_preference_steering import (
    plot_convergence_curves,
    plot_preference_alignment,
    generate_experiment_report,
    compare_feature_space_report,
    compare_approaches,
)


def _git_commit() -> str:
    """Best-effort current git commit hash for run provenance (None on failure)."""
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return None


def run_full_experiment(config: PreferenceExperimentConfig) -> None:
    """
    Run the complete preference steering experiment:
    1. Download and prepare data
    2. Train shared theta on synthetic users
    3. Test on held-out users (fixed theta mode)
    4. Test on held-out users (adaptive theta mode)
    5. Evaluate and visualize results
    """
    print("\n" + "=" * 70)
    print("PREFERENCE STEERING EXPERIMENT")
    print("=" * 70)

    # Step 1: Data preparation
    print("\n[STEP 1] DATA PREPARATION")
    print("-" * 70)

    images_dir = Path(f"{config.sneaker_data_dir}/images/shoes_100")

    # Check if images already exist
    if images_dir.exists() and len(list(images_dir.glob("*.jpg"))) >= 100:
        print(f"✓ Found {len(list(images_dir.glob('*.jpg')))} images in {images_dir}")
        print("Skipping download (images already present)")
    else:
        print("Downloading Zappos dataset...")
        download_zappos_dataset(output_dir=f"{config.sneaker_data_dir}/images")

    features_dir = f"{config.sneaker_data_dir}/features"
    features_file = f"{features_dir}/feature_labels.json"
    embeddings_file = f"{features_dir}/image_embeddings.json"
    stats_file = f"{features_dir}/concept_stats.json"

    # (Re)label if any cache is missing or in an older format. concept_stats.json
    # only exists for the current debiased-softmax f(x), so its absence forces a
    # re-label of caches produced by earlier versions.
    needs_labeling = not (
        Path(features_file).exists()
        and Path(embeddings_file).exists()
        and Path(stats_file).exists()
    )
    if not needs_labeling:
        try:
            with open(features_file) as f:
                sample = next(iter(json.load(f).values()))
            needs_labeling = "concept_scores" not in sample
        except (StopIteration, json.JSONDecodeError):
            needs_labeling = True

    if needs_labeling:
        print("Labeling features with CLIP (concepts + embeddings)...")
        label_features_with_clip(
            image_dir=f"{config.sneaker_data_dir}/images/shoes_100",
            output_dir=features_dir,
            feature_concepts=config.feature_concepts,
        )
    else:
        print(f"✓ Found feature caches in {features_dir}")
        print("Skipping feature extraction (caches already present)")

    print("Validating concept feature space...")
    validate_feature_space(
        image_dir=f"{config.sneaker_data_dir}/images/shoes_100",
        labels_file=features_file,
    )

    # Decide which feature spaces to evaluate
    spaces = (
        config.feature_spaces
        if config.compare_feature_spaces
        else [config.feature_space]
    )
    print(f"\nFeature spaces to evaluate: {spaces}")

    # Optionally reuse a previously exported bundle (LoRA + trained theta) instead
    # of retraining. The LoRA path and the preference model come from the bundle.
    preloaded_oracle = None
    if config.load_bundle:
        import torch
        from preference.utils.hf_artifacts import load_preference_bundle

        device = "cuda" if torch.cuda.is_available() else "cpu"
        bundle = load_preference_bundle(config.load_bundle, device=device)
        preloaded_oracle = bundle["oracle"]
        if not config.base_only and not config.lora_checkpoint and bundle["lora_dir"]:
            config.lora_checkpoint = bundle["lora_dir"]
        print(
            f"[load-bundle] theta loaded from {config.load_bundle}; "
            f"LoRA = {config.lora_checkpoint}"
        )

    # Load the diffusion pipeline once (expensive) and reuse it across spaces.
    diffusion = build_diffusion_bundle(config) if config.use_real_diffusion else None

    # Durable per-example/per-step metrics DB (one row committed as produced).
    metrics_db = None
    if config.log_metrics_db:
        from dataclasses import asdict
        from preference.utils.metrics_db import MetricsDB

        metrics_db = MetricsDB(f"{config.results_dir}/metrics.sqlite")
        run_id = metrics_db.start_run(
            config=asdict(config),
            base_model=config.diffusion_model_id,
            lora=config.lora_checkpoint,
            use_real_diffusion=config.use_real_diffusion,
            git_commit=_git_commit(),
        )
        print(f"[metrics] logging to {config.results_dir}/metrics.sqlite (run {run_id})")

    try:
        summary_by_space = {}
        for feature_space in spaces:
            # Keep each space's artifacts separate when comparing.
            space_results_dir = (
                str(Path(config.results_dir) / feature_space)
                if len(spaces) > 1
                else config.results_dir
            )
            results_by_mode = run_single_space(
                config,
                feature_space,
                features_file,
                space_results_dir,
                diffusion,
                preloaded_oracle=preloaded_oracle,
                metrics_db=metrics_db,
            )
            summary_by_space[feature_space] = {
                name: res["aggregate"] for name, res in results_by_mode.items()
            }

        # Cross-space comparison
        if len(spaces) > 1:
            compare_feature_space_report(summary_by_space, config.results_dir)
    finally:
        if metrics_db is not None:
            metrics_db.close()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {config.results_dir}")


def build_diffusion_bundle(config: PreferenceExperimentConfig) -> dict:
    """Load the SDXL(+LoRA) pipeline, feature extractor and steering config once.

    Returned bundle is reused for every feature space and both test modes.
    """
    import torch
    from models.model_diffusion import DiffusionGeneratorLoRA
    from preference.models.feature_extractor import DifferentiableFeatureExtractor
    from preference.inference.preference_steering import PreferenceSteeringConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nLoading diffusion pipeline for real preference-steered generation...")
    diff = DiffusionGeneratorLoRA(model_id=config.diffusion_model_id, device=device)
    if config.base_only:
        print(
            f"[base-only] steering on vanilla {config.diffusion_model_id} "
            "(no LoRA), as requested."
        )
    elif config.lora_checkpoint:
        print(f"Applying trained LoRA from {config.lora_checkpoint}")
        # Native diffusers round-trip: weights saved with save_lora_weights load
        # directly with load_lora_weights (no setup_lora adapter needed first).
        diff.load_lora(config.lora_checkpoint)
    else:
        print(
            "WARNING: no lora_checkpoint set -- using base model (generic sneakers, "
            "not your fine-tuned ones). Pass --no-lora to make this intentional."
        )

    feature_extractor = DifferentiableFeatureExtractor(
        feature_concepts=config.feature_concepts,
        device=device,
        concept_stats_file=f"{config.sneaker_data_dir}/features/concept_stats.json",
    )
    steering_config = PreferenceSteeringConfig(
        steering_strength=config.steering_strength,
        steer_every_k_steps=config.steer_every_k_steps,
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
        max_grad_norm=config.max_grad_norm,
        image_size=config.image_size,
        debug_log=config.debug_log_steering,
    )
    return {
        "pipeline": diff.pipeline,
        "feature_extractor": feature_extractor,
        "steering_config": steering_config,
        "device": device,
    }


def run_single_space(
    config: PreferenceExperimentConfig,
    feature_space: str,
    features_file: str,
    results_dir: str,
    diffusion: dict = None,
    preloaded_oracle=None,
    metrics_db=None,
) -> dict:
    """Run training + the selected test mode(s) + evaluation for one feature space.

    Returns a dict mapping each run mode name ("nosteer"/"fixed"/"adaptive") to its
    results dict.
    """
    print("\n" + "=" * 70)
    print(f"FEATURE SPACE: {feature_space.upper()}")
    print("=" * 70)

    # Step 2: Training phase (skipped when reusing a loaded bundle's theta)
    if preloaded_oracle is not None:
        print("\n[STEP 2] TRAINING PHASE -- skipped (theta loaded from bundle)")
        trained_oracle = preloaded_oracle
    else:
        print("\n[STEP 2] TRAINING PHASE")
        print("-" * 70)
        trained_oracle, training_metadata = simulate_training_phase(
            num_synthetic_users=config.num_synthetic_train_users,
            interactions_per_user=config.interactions_per_train_user,
            theta_learning_rate=config.theta_learning_rate,
            features_file=features_file,
            output_dir=results_dir,
            feature_space=feature_space,
        )

    # Build the steered generator on top of this space's trained oracle.
    generator = None
    feature_extractor = None
    steering_config = None
    if diffusion is not None:
        from preference.inference.preference_oracle_inference import (
            PreferenceOracleInference,
        )
        from preference.inference.preference_steering import PreferenceSteeredGenerator

        feature_extractor = diffusion["feature_extractor"]
        steering_config = diffusion["steering_config"]
        oracle_inf = PreferenceOracleInference(
            trained_oracle, feature_extractor, device=diffusion["device"]
        )
        generator = PreferenceSteeredGenerator(
            diffusion["pipeline"], oracle_inf, device=diffusion["device"]
        )

    test_kwargs = dict(
        generator=generator,
        feature_extractor=feature_extractor,
        steering_config=steering_config,
        generation_prompt=config.generation_prompt,
        metrics_db=metrics_db,
    )

    # Step 3: Test phase. Each configuration is one (adapt_theta, steer) setting.
    #   nosteer  -- base model + LoRA, NO preference steering (the baseline control)
    #   fixed    -- steered generation, theta frozen
    #   adaptive -- steered generation, theta keeps updating
    # Default runs all three so the baseline-vs-steered comparison is produced.
    # `--only` picks a single configuration -- e.g. `--only nosteer` for just the
    # base-model baseline. adaptive runs last because it mutates theta in place.
    mode_specs = {
        "nosteer": dict(adapt_theta=False, steer=False),
        "fixed": dict(adapt_theta=False, steer=True),
        "adaptive": dict(adapt_theta=True, steer=True),
    }
    selected = (
        [config.only_mode] if config.only_mode else ["nosteer", "fixed", "adaptive"]
    )
    # Tag the on-disk results/images so base-SD1.5 and LoRA runs coexist in the
    # SAME results dir (instead of overwriting) and all show up in the comparison.
    variant = "base" if config.base_only else ""

    results_by_mode: dict[str, dict] = {}
    for i, name in enumerate(selected):
        spec = mode_specs[name]
        print(f"\n[STEP 3.{i + 1}] TEST PHASE - {name.upper()}")
        print("-" * 70)
        results_by_mode[name] = simulate_test_phase(
            trained_oracle=trained_oracle,
            num_test_users=config.num_synthetic_test_users,
            rounds_per_user=config.rounds_per_test_user,
            features_file=features_file,
            adapt_theta=spec["adapt_theta"],
            steer=spec["steer"],
            variant=variant,
            user_learning_rate=config.user_learning_rate,
            theta_learning_rate=config.theta_learning_rate,
            output_dir=results_dir,
            feature_space=feature_space,
            **test_kwargs,
        )

    # Step 4: Evaluation. The two-mode comparison plots need both fixed and
    # adaptive; for a single-config run we just report that config's aggregate
    # (full per-example metrics are in the result JSON and metrics.sqlite).
    print("\n[STEP 4] EVALUATION & VISUALIZATION")
    print("-" * 70)
    results_fixed = results_by_mode.get("fixed")
    results_adaptive = results_by_mode.get("adaptive")
    if results_fixed and results_adaptive:
        plot_convergence_curves(results_fixed, results_adaptive, results_dir)
        plot_preference_alignment(results_fixed, results_adaptive, results_dir)
        generate_experiment_report(results_fixed, results_adaptive, results_dir)
    for name, res in results_by_mode.items():
        agg = res["aggregate"]
        print(
            f"  [{name}] mean click rate = {agg['mean_click_rate']:.3f} "
            f"(±{agg['std_click_rate']:.3f}), "
            f"mean alignment = {agg['mean_alignment']:.3f}"
        )

    # Compare every solution sharing this results dir (baseline, LoRA-steered,
    # base-SD1.5-steered, ...) -- scans all test_results_*.json on disk, so runs
    # done separately (e.g. via --only or --no-lora) all show up together.
    compare_approaches(
        output_dir=results_dir,
        real_image_dir=f"{config.sneaker_data_dir}/images/shoes_100",
        device=diffusion["device"] if diffusion else None,
    )

    # Step 5: Export a reusable bundle (LoRA + theta + stats) for HF sync
    if config.export_dir:
        from preference.utils.hf_artifacts import export_preference_bundle, push_to_hf

        export_dir = config.export_dir
        if config.compare_feature_spaces:
            export_dir = str(Path(export_dir) / feature_space)
        manifest = {
            "base_model": config.diffusion_model_id,
            "image_size": config.image_size,
            "feature_space": feature_space,
            "feature_concepts": config.feature_concepts,
            "generation_prompt": config.generation_prompt,
            "num_features": trained_oracle.num_features,
        }
        export_preference_bundle(
            export_dir=export_dir,
            lora_dir=config.lora_checkpoint,
            oracle=trained_oracle,
            concept_stats_file=f"{config.sneaker_data_dir}/features/concept_stats.json",
            manifest=manifest,
            results_dir=results_dir,
        )
        if config.hf_sync_target:
            push_to_hf(export_dir, config.hf_sync_target)

    return results_by_mode


def main():
    parser = argparse.ArgumentParser(
        description="Run preference steering experiment"
    )
    parser.add_argument(
        "--num-train-users",
        type=int,
        default=50,
        help="Number of synthetic users for training",
    )
    parser.add_argument(
        "--interactions-per-user",
        type=int,
        default=100,
        help="Interactions per training user",
    )
    parser.add_argument(
        "--num-test-users",
        type=int,
        default=3,
        help="Number of held-out test users",
    )
    parser.add_argument(
        "--rounds-per-test-user",
        type=int,
        default=50,
        help="Generation rounds per test user",
    )
    parser.add_argument(
        "--theta-lr",
        type=float,
        default=0.01,
        help="Learning rate for shared theta",
    )
    parser.add_argument(
        "--user-lr", type=float, default=0.05, help="Learning rate for u_A"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="preference/results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Path to trained LoRA weights (local dir) for sneaker generation",
    )
    parser.add_argument(
        "--no-real-diffusion",
        action="store_true",
        help="Use the feature-space proxy instead of running the diffusion model",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Steer the vanilla base SD1.5 model with NO LoRA (overrides --lora). "
        "Inference runs preference steering on plain SD1.5.",
    )
    parser.add_argument(
        "--only",
        choices=["nosteer", "fixed", "adaptive"],
        default=None,
        help="Run only one evaluation configuration instead of the default "
        "fixed + adaptive comparison. 'nosteer' is the base-model baseline with "
        "no preference steering.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny validation run: 1 user, 3 rounds, 15 steps, step logging on",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a sneaker",
        help="Text prompt used for generation",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="After the run, write a reusable bundle (LoRA + preference model) here",
    )
    parser.add_argument(
        "--hf-sync-target",
        type=str,
        default=None,
        help="If set, run `hf sync <export-dir> <target>` "
        "(e.g. hf://buckets/tkher/preferencemodels)",
    )
    parser.add_argument(
        "--load-bundle",
        type=str,
        default=None,
        help="Load LoRA + theta from a previously exported/pulled bundle dir; "
        "skips theta training",
    )

    args = parser.parse_args()

    # If a sync target is given but no export dir, default to a sensible location.
    export_dir = args.export_dir
    if args.hf_sync_target and not export_dir:
        export_dir = "preference/exports/preferencemodels"

    # --no-lora forces base SD1.5 even if --lora was passed.
    lora_checkpoint = None if args.no_lora else args.lora

    # Create config
    config = PreferenceExperimentConfig(
        num_synthetic_train_users=args.num_train_users,
        interactions_per_train_user=args.interactions_per_user,
        num_synthetic_test_users=args.num_test_users,
        rounds_per_test_user=args.rounds_per_test_user,
        theta_learning_rate=args.theta_lr,
        user_learning_rate=args.user_lr,
        results_dir=args.results_dir,
        lora_checkpoint=lora_checkpoint,
        base_only=args.no_lora,
        use_real_diffusion=not args.no_real_diffusion,
        generation_prompt=args.prompt,
        export_dir=export_dir,
        hf_sync_target=args.hf_sync_target,
        load_bundle=args.load_bundle,
        only_mode=args.only,
    )

    # Smoke test: keep generation cheap so you can confirm it works end-to-end.
    if args.smoke:
        config.num_synthetic_test_users = 1
        config.rounds_per_test_user = 3
        config.num_inference_steps = 15
        config.debug_log_steering = True
        print("[smoke] 1 test user, 3 rounds, 15 inference steps, step logging on")

    # Run experiment
    run_full_experiment(config)


if __name__ == "__main__":
    main()
