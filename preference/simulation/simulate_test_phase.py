from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from preference.simulation.synthetic_user_simulator import create_diverse_synthetic_users
from preference.simulation.simulate_training_phase import load_image_features


def simulate_test_phase(
    trained_oracle,
    num_test_users: int = 3,
    rounds_per_user: int = 50,
    features_file: str = "sneakerdata/features/feature_labels.json",
    adapt_theta: bool = False,
    steer: bool = True,
    variant: str = "",
    user_learning_rate: float = 0.05,
    theta_learning_rate: float = 0.01,
    output_dir: str = "preference/results",
    seed: int = 42,
    feature_space: str = "concepts",
    generator=None,
    feature_extractor=None,
    steering_config=None,
    generation_prompt: str = "a sneaker",
    metrics_db=None,
) -> dict:
    """
    Simulate test phase: evaluate on held-out users with diffusion steering.

    This simulates:
    - User 51, 52, 53 (never seen during training)
    - For each user: collect initial clicks to extract u_A
    - Then generate images with diffusion steering (using theta and u_A)
    - Measure click rates and convergence

    Args:
        trained_oracle: PreferenceOracle trained on 50 users
        num_test_users: number of held-out users to test
        rounds_per_user: number of generation rounds per user
        features_file: path to feature labels
        adapt_theta: whether to update theta from test user interactions
        user_learning_rate: learning rate for u_A updates
        theta_learning_rate: learning rate for theta updates (if adapt_theta=True)
        output_dir: directory to save results
        seed: random seed

    Returns:
        results dictionary with metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Real diffusion generation is active only when a steered generator and a
    # feature extractor are supplied; otherwise we fall back to the lightweight
    # feature-space proxy.
    use_real = generator is not None and feature_extractor is not None

    # Short tag used for the mode label, image filenames and result JSON.
    # `steer=False` is the no-steering baseline: the base model + LoRA generate
    # exactly as trained, with no preference gradient applied -- the control the
    # steered modes are measured against. `variant` keeps distinct solutions
    # (e.g. base-SD1.5 vs LoRA) separate within the SAME results dir, so the
    # comparison picks all of them up instead of overwriting each other.
    mode_tag = "nosteer" if not steer else ("adaptive" if adapt_theta else "fixed")
    if variant:
        mode_tag = f"{mode_tag}_{variant}"

    # When not steering, run a plain text-to-image pass (steer_every_k_steps=0
    # disables the gradient block in PreferenceSteeredGenerator.generate).
    gen_steering_config = steering_config
    if use_real and not steer and steering_config is not None:
        import dataclasses

        gen_steering_config = dataclasses.replace(steering_config, steer_every_k_steps=0)

    images_dir = output_path / "generated_images"
    if use_real:
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"[real diffusion] saving generated images under {images_dir} (mode={mode_tag})")

    # Load image features
    print(f"Loading image features (feature_space={feature_space})...")
    image_ids, features_list = load_image_features(
        features_file, feature_space=feature_space
    )

    # Create held-out test users (IDs 51, 52, 53), anchored to real shoes
    print(f"Creating {num_test_users} held-out synthetic users...")
    test_users = create_diverse_synthetic_users(
        num_users=num_test_users,
        num_features=features_list[0].shape[0],
        seed=seed + 100,  # Different seed for diversity
        feature_matrix=features_list,
    )

    num_features = features_list[0].shape[0]
    mode_label = (
        "baseline_nosteer"
        if not steer
        else ("adaptive_theta" if adapt_theta else "fixed_theta")
    )
    if variant:
        mode_label = f"{mode_label}_{variant}"

    # Human-readable name used in the comparison report/plots.
    variant_disp = {"": "", "base": "base SD1.5", "lora": "LoRA"}.get(variant, variant)
    if not steer:
        display = "No steering (baseline)" + (f" [{variant_disp}]" if variant else "")
    else:
        _base = "Adaptive theta" if adapt_theta else "Fixed theta"
        display = f"{_base} (steered{', ' + variant_disp if variant else ''})"

    results = {
        "mode": mode_label,
        "mode_tag": mode_tag,
        "variant": variant,
        "display": display,
        "steer": steer,
        "test_users": [],
    }

    print("\n--- TEST PHASE ---")
    print(f"Mode: {display}\n")

    for test_user in test_users:
        print(f"\n{'='*60}")
        print(f"Testing {test_user}")
        print(f"{'='*60}")

        user_result = {
            "user_id": test_user.user_id,
            "true_preference": test_user.true_preference.tolist(),
            "rounds": [],
        }

        # Phase 1: Collect initial clicks to bootstrap u_A
        print(f"\nPhase 1: Collecting initial signal ({min(10, len(features_list))} images)...")
        initial_clicks = []
        initial_skips = []

        for i in range(min(10, len(features_list))):
            features = features_list[i]
            clicked = test_user.decide_click(features, add_noise=False)
            if clicked:
                initial_clicks.append(features.clone())
            else:
                initial_skips.append(features.clone())

            if metrics_db is not None:
                metrics_db.log_example(
                    feature_space=feature_space,
                    mode=results["mode"],
                    phase="bootstrap",
                    user_id=test_user.user_id,
                    round=i,
                    clicked=clicked,
                    image_id=image_ids[i] if i < len(image_ids) else None,
                    f=features.detach().cpu(),
                    true_pref=test_user.true_preference,
                )

        # Extract user representation u_A
        if initial_clicks:
            u_A = torch.stack(initial_clicks).mean(dim=0)
        else:
            u_A = torch.ones(num_features) / num_features

        u_A = u_A / (u_A.norm() + 1e-8)
        print(
            f"Initial u_A extracted: {u_A.tolist()} (from {len(initial_clicks)} clicks)"
        )

        # Phase 2: Generate personalized images and measure click rate
        print(f"\nPhase 2: Generating {rounds_per_user} personalized images...")

        for round_idx in range(rounds_per_user):
            image_path = None

            if use_real:
                # Real diffusion: steer generation toward (theta + u_A) and use
                # the GENERATED image's measured features for the click decision.
                step_logger = None
                if metrics_db is not None:
                    def step_logger(**kw):
                        metrics_db.log_step(
                            feature_space=feature_space,
                            mode=results["mode"],
                            user_id=test_user.user_id,
                            round=round_idx,
                            **kw,
                        )
                image = generator.generate(
                    generation_prompt,
                    gen_steering_config,
                    user_preferences=u_A,
                    step_logger=step_logger,
                )
                image_path = images_dir / (
                    f"user{test_user.user_id}_"
                    f"{mode_tag}_"
                    f"round{round_idx:03d}.png"
                )
                image.save(image_path)
                steered_features = (
                    feature_extractor.extract_features_from_pil(image)
                    .detach()
                    .cpu()
                )
            else:
                # Feature-space proxy (no image).
                random_features = torch.randn(num_features)
                random_features = random_features / (random_features.norm() + 1e-8)
                if not steer:
                    # Baseline: base-model output is independent of preferences,
                    # so the proxy is pure random exploration.
                    steered_features = random_features
                else:
                    # Generated features blend the preference direction with
                    # random exploration.
                    theta = trained_oracle.get_theta().cpu()
                    combined_preference = (theta + u_A) / 2
                    steered_features = (
                        0.7 * combined_preference + 0.3 * random_features
                    )  # 70% preference-guided, 30% random
                    steered_features = steered_features / (
                        steered_features.norm() + 1e-8
                    )

            # User clicks based on how well the generated shoe aligns with their
            # true preference
            clicked = test_user.decide_click(steered_features, add_noise=True)

            # Log every generated shoe
            feat_str = "[" + ", ".join(f"{v:.3f}" for v in steered_features.tolist()) + "]"
            saved = f" -> {image_path.name}" if image_path is not None else ""
            print(
                f"  [gen] user {test_user.user_id} round {round_idx + 1}/{rounds_per_user} "
                f"f(x)={feat_str} clicked={clicked}{saved}"
            )

            # Record round
            round_result = {
                "round": round_idx,
                "steered_features": steered_features.tolist(),
                "user_true_pref": test_user.true_preference.tolist(),
                "user_learned_pref": u_A.tolist(),
                "clicked": clicked,
                "image_path": str(image_path) if image_path is not None else None,
            }
            user_result["rounds"].append(round_result)

            # Record full per-example metrics to the durable DB (u_A and theta
            # here are the values that produced this image, before the update).
            if metrics_db is not None:
                theta_now = trained_oracle.get_theta().cpu()
                true_pref = test_user.true_preference
                theta_dot_f = float(torch.dot(theta_now, steered_features))
                uA_dot_f = float(torch.dot(u_A, steered_features))
                alignment = float(
                    torch.nn.functional.cosine_similarity(
                        u_A.unsqueeze(0), true_pref.unsqueeze(0)
                    )
                )
                cum_click_rate = sum(r["clicked"] for r in user_result["rounds"]) / (
                    round_idx + 1
                )
                metrics_db.log_example(
                    feature_space=feature_space,
                    mode=results["mode"],
                    phase="generation",
                    user_id=test_user.user_id,
                    round=round_idx,
                    clicked=clicked,
                    pref_score=theta_dot_f + uA_dot_f,
                    theta_dot_f=theta_dot_f,
                    uA_dot_f=uA_dot_f,
                    alignment_uA_true=alignment,
                    cum_click_rate=cum_click_rate,
                    image_path=str(image_path) if image_path is not None else None,
                    f=steered_features,
                    uA=u_A,
                    theta=theta_now,
                    true_pref=true_pref,
                )

            # Update u_A and optionally theta
            if clicked:
                gradient = steered_features
            else:
                gradient = -steered_features

            # Update u_A
            u_A = u_A + user_learning_rate * gradient
            u_A = u_A / (u_A.norm() + 1e-8)

            # Update theta if adaptive mode
            if adapt_theta:
                trained_oracle.update_theta(gradient, learning_rate=theta_learning_rate)

            if (round_idx + 1) % max(10, rounds_per_user // 5) == 0:
                click_rate = (
                    sum(r["clicked"] for r in user_result["rounds"])
                    / (round_idx + 1)
                )
                alignment = torch.nn.functional.cosine_similarity(
                    u_A.unsqueeze(0),
                    test_user.true_preference.unsqueeze(0),
                )
                print(
                    f"  Round {round_idx+1}/{rounds_per_user} | "
                    f"Click Rate: {100*click_rate:.1f}% | "
                    f"Alignment with true pref: {alignment.item():.3f}"
                )

        # Compute final metrics
        clicks = sum(r["clicked"] for r in user_result["rounds"])
        final_click_rate = clicks / rounds_per_user

        final_alignment = torch.nn.functional.cosine_similarity(
            u_A.unsqueeze(0),
            test_user.true_preference.unsqueeze(0),
        ).item()

        user_result["final_click_rate"] = final_click_rate
        user_result["final_u_A"] = u_A.tolist()
        user_result["final_alignment"] = final_alignment

        print(f"\nFinal Results for {test_user}:")
        print(f"  Click Rate: {100*final_click_rate:.1f}%")
        print(f"  Preference Alignment: {final_alignment:.3f}")
        print(f"  Final u_A: {u_A.tolist()}")

        results["test_users"].append(user_result)

    # Aggregate metrics
    all_click_rates = [u["final_click_rate"] for u in results["test_users"]]
    all_alignments = [u["final_alignment"] for u in results["test_users"]]

    results["aggregate"] = {
        "mean_click_rate": float(np.mean(all_click_rates)),
        "std_click_rate": float(np.std(all_click_rates)),
        "mean_alignment": float(np.mean(all_alignments)),
        "std_alignment": float(np.std(all_alignments)),
    }

    # Save results
    results_file = output_path / f"test_results_{results['mode']}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nTest phase complete!")
    print(f"Results saved to {results_file}")
    print(f"\nAggregate Results:")
    print(
        f"  Mean Click Rate: {100*results['aggregate']['mean_click_rate']:.1f}% "
        f"(±{100*results['aggregate']['std_click_rate']:.1f}%)"
    )
    print(
        f"  Mean Alignment: {results['aggregate']['mean_alignment']:.3f} "
        f"(±{results['aggregate']['std_alignment']:.3f})"
    )

    return results
