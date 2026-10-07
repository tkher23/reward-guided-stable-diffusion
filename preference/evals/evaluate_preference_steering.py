from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence_curves(
    results_fixed_theta: dict,
    results_adaptive_theta: dict,
    output_dir: str = "preference/results",
) -> None:
    """
    Plot convergence curves comparing fixed vs. adaptive theta modes.

    Args:
        results_fixed_theta: results dict from simulate_test_phase (fixed mode)
        results_adaptive_theta: results dict from simulate_test_phase (adaptive mode)
        output_dir: where to save figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract click rates over rounds for each user
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Individual user trajectories
    ax = axes[0]
    for mode_name, results in [
        ("Fixed θ", results_fixed_theta),
        ("Adaptive θ", results_adaptive_theta),
    ]:
        for user_result in results["test_users"]:
            user_id = user_result["user_id"]
            click_trajectory = [r["clicked"] for r in user_result["rounds"]]

            # Compute moving average (window=5)
            window = 5
            moving_avg = (
                np.convolve(
                    click_trajectory, np.ones(window) / window, mode="valid"
                )
                * 100
            )

            ax.plot(
                range(len(moving_avg)),
                moving_avg,
                label=f"{mode_name} User {user_id}",
                alpha=0.7,
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Click Rate (% moving avg, window=5)")
    ax.set_title("User Click Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Aggregate comparison
    ax = axes[1]
    modes = ["Fixed θ", "Adaptive θ"]
    means = [
        results_fixed_theta["aggregate"]["mean_click_rate"] * 100,
        results_adaptive_theta["aggregate"]["mean_click_rate"] * 100,
    ]
    stds = [
        results_fixed_theta["aggregate"]["std_click_rate"] * 100,
        results_adaptive_theta["aggregate"]["std_click_rate"] * 100,
    ]

    ax.bar(modes, means, yerr=stds, capsize=10, alpha=0.7, color=["blue", "green"])
    ax.set_ylabel("Mean Click Rate (%)")
    ax.set_title("Aggregate Performance Comparison")
    ax.set_ylim([0, 100])
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 2, f"{m:.1f}%", ha="center")

    plt.tight_layout()
    fig_path = output_path / "convergence_comparison.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")
    plt.close()


def plot_preference_alignment(
    results_fixed_theta: dict,
    results_adaptive_theta: dict,
    output_dir: str = "preference/results",
) -> None:
    """Plot how well u_A converges to true user preferences."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for mode_name, results in [
        ("Fixed θ", results_fixed_theta),
        ("Adaptive θ", results_adaptive_theta),
    ]:
        alignments = [u["final_alignment"] for u in results["test_users"]]
        user_ids = [u["user_id"] for u in results["test_users"]]

        ax.plot(
            user_ids,
            alignments,
            marker="o",
            label=mode_name,
            linewidth=2,
            markersize=10,
        )

    ax.set_xlabel("User ID")
    ax.set_ylabel("Cosine Similarity (u_A vs True Preference)")
    ax.set_title("Preference Learning Alignment")
    ax.set_ylim([-1, 1])
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_path / "preference_alignment.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")
    plt.close()


def generate_experiment_report(
    results_fixed_theta: dict,
    results_adaptive_theta: dict,
    output_dir: str = "preference/results",
) -> str:
    """Generate a text report comparing the two modes."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "=" * 70,
        "PREFERENCE STEERING EXPERIMENT REPORT",
        "=" * 70,
        "",
        "SUMMARY",
        "-" * 70,
        f"Fixed Theta Mode:",
        f"  Mean Click Rate:    {100*results_fixed_theta['aggregate']['mean_click_rate']:.1f}% "
        f"(±{100*results_fixed_theta['aggregate']['std_click_rate']:.1f}%)",
        f"  Mean Alignment:     {results_fixed_theta['aggregate']['mean_alignment']:.3f} "
        f"(±{results_fixed_theta['aggregate']['std_alignment']:.3f})",
        "",
        f"Adaptive Theta Mode:",
        f"  Mean Click Rate:    {100*results_adaptive_theta['aggregate']['mean_click_rate']:.1f}% "
        f"(±{100*results_adaptive_theta['aggregate']['std_click_rate']:.1f}%)",
        f"  Mean Alignment:     {results_adaptive_theta['aggregate']['mean_alignment']:.3f} "
        f"(±{results_adaptive_theta['aggregate']['std_alignment']:.3f})",
        "",
        f"Improvement (Adaptive vs Fixed):",
        f"  Click Rate Delta:   {(100*results_adaptive_theta['aggregate']['mean_click_rate'] - 100*results_fixed_theta['aggregate']['mean_click_rate']):.1f}% points",
        f"  Alignment Delta:    {(results_adaptive_theta['aggregate']['mean_alignment'] - results_fixed_theta['aggregate']['mean_alignment']):.3f}",
        "",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    print(report_text)

    report_file = output_path / "experiment_report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to {report_file}")
    return report_text


def compare_feature_space_report(
    summary_by_space: dict,
    output_dir: str = "preference/results",
) -> str:
    """Compare feature spaces (e.g. concepts vs embedding) side by side.

    Args:
        summary_by_space: {space_name: {"fixed": aggregate, "adaptive": aggregate}}
            where each aggregate has mean/std click_rate and alignment.
        output_dir: where to save the report and figure.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    spaces = list(summary_by_space.keys())

    lines = [
        "=" * 78,
        "FEATURE SPACE COMPARISON",
        "=" * 78,
        "",
        f"{'Feature space':<14} {'Mode':<10} {'Click rate':>14} {'Alignment':>16}",
        "-" * 78,
    ]
    for space in spaces:
        for mode in ("fixed", "adaptive"):
            agg = summary_by_space[space][mode]
            lines.append(
                f"{space:<14} {mode:<10} "
                f"{100*agg['mean_click_rate']:>7.1f}% (±{100*agg['std_click_rate']:>4.1f}) "
                f"{agg['mean_alignment']:>8.3f} (±{agg['std_alignment']:>5.3f})"
            )
        lines.append("-" * 78)
    lines.append("")
    lines.append("=" * 78)

    report_text = "\n".join(lines)
    print(report_text)

    report_file = output_path / "feature_space_comparison.txt"
    with open(report_file, "w") as f:
        f.write(report_text)
    print(f"\nComparison report saved to {report_file}")

    # Grouped bar chart: click rate and alignment per (space, mode)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(spaces))
    width = 0.35

    for ax, (metric, label, scale) in zip(
        axes,
        [
            ("mean_click_rate", "Mean Click Rate (%)", 100.0),
            ("mean_alignment", "Mean Alignment (cosine)", 1.0),
        ],
    ):
        std_key = "std_click_rate" if metric == "mean_click_rate" else "std_alignment"
        for offset, mode, color in [
            (-width / 2, "fixed", "tab:blue"),
            (width / 2, "adaptive", "tab:green"),
        ]:
            means = [summary_by_space[s][mode][metric] * scale for s in spaces]
            errs = [summary_by_space[s][mode][std_key] * scale for s in spaces]
            ax.bar(
                x + offset, means, width, yerr=errs, capsize=6,
                label=f"{mode} θ", color=color, alpha=0.8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(spaces)
        ax.set_ylabel(label)
        ax.set_title(label.split(" (")[0])
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = output_path / "feature_space_comparison.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")
    plt.close()

    return report_text


def _realism_by_mode(
    real_image_dir: str,
    generated_images_dir: str,
    tags: list[str],
    device: str | None = None,
) -> dict:
    """Image realism per mode, reusing the InceptionV3 FID metric from
    `evals/compute_fid_metrics.py` (outside the preference package).

    For each mode it computes the FID between that mode's generated images and
    the real sneaker images. Falls back to mean InceptionV3 distance-to-real-
    centroid when the covariance is too ill-conditioned for a finite FID (common
    with few samples). Returns {tag: {"score", "method", "n_generated"}} or {} if
    the metric can't run (e.g. a proxy run that saved no images).
    """
    from PIL import Image

    real_paths = sorted(glob.glob(str(Path(real_image_dir) / "*.jpg")))
    if not real_paths:
        print(f"[compare] no real images in {real_image_dir}; skipping realism metric")
        return {}
    # Only proceed if at least one mode actually produced images.
    if not any(
        glob.glob(str(Path(generated_images_dir) / f"*_{t}_round*.png")) for t in tags
    ):
        return {}

    try:
        import torch
        from evals.compute_fid_metrics import FIDCalculator
    except Exception as e:  # torch/torchvision/scipy missing, etc.
        print(f"[compare] realism metric unavailable ({e}); skipping FID")
        return {}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fid = FIDCalculator(device=device)

    def feats(paths):
        return np.array(
            [fid.extract_features(Image.open(p).convert("RGB")) for p in paths]
        )

    real_feats = feats(real_paths)
    real_centroid = real_feats.mean(axis=0)

    out = {}
    for tag in tags:
        gen_paths = sorted(
            glob.glob(str(Path(generated_images_dir) / f"*_{tag}_round*.png"))
        )
        if not gen_paths:
            continue
        gen_feats = feats(gen_paths)
        method = "fid"
        try:
            score = fid.compute_fid(real_feats, gen_feats)
            if not np.isfinite(score):
                raise ValueError("non-finite FID")
        except Exception:
            score = float(
                np.mean([np.linalg.norm(g - real_centroid) for g in gen_feats])
            )
            method = "centroid_l2"
        out[tag] = {
            "score": float(score),
            "method": method,
            "n_generated": len(gen_paths),
        }
        print(f"[compare] realism {tag}: {score:.3f} ({method}, n={len(gen_paths)})")
    return out


def _approach_sort_key(item):
    """Order approaches: baselines first, then LoRA variant before others,
    fixed before adaptive."""
    tag, d = item
    steer = bool(d.get("steer", True))
    variant = d.get("variant", "")
    adaptive = "adaptive" in str(d.get("mode", tag))
    return (1 if steer else 0, 1 if variant else 0, 1 if adaptive else 0, tag)


def compare_approaches(
    output_dir: str = "preference/results",
    real_image_dir: str | None = None,
    device: str | None = None,
) -> dict | None:
    """Compare every solution found in `output_dir` against each other.

    Scans all `test_results_*.json` in the results dir -- so the no-steer
    baseline, the LoRA-steered modes (fixed/adaptive) and the base-SD1.5-steered
    modes all show up together, however many runs produced them, as long as they
    share this one results dir. Reports click-through rate, preference alignment
    (efficacy) and, when real + generated images are available, image realism
    (reused InceptionV3 FID). Writes:
      - approach_comparison.txt   (table + deltas vs the baseline)
      - approach_comparison.png   (grouped bars)
      - approach_comparison.json  (machine-readable)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover every saved solution in this results dir.
    found = {}
    for f in sorted(output_path.glob("test_results_*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "aggregate" not in data:
            continue
        tag = data.get("mode_tag") or f.stem[len("test_results_"):]
        found[tag] = data

    if len(found) < 2:
        print("[compare] fewer than 2 approaches available; skipping comparison")
        return None

    ordered_items = sorted(found.items(), key=_approach_sort_key)
    ordered = [t for t, _ in ordered_items]
    print(f"[compare] comparing {len(ordered)} approaches: {', '.join(ordered)}")

    realism = {}
    if real_image_dir:
        realism = _realism_by_mode(
            real_image_dir, str(output_path / "generated_images"), ordered, device
        )

    # Assemble per-approach metrics.
    rows = {}
    for tag in ordered:
        d = found[tag]
        agg = d["aggregate"]
        rows[tag] = {
            "display": d.get("display", tag),
            "mean_click_rate": agg["mean_click_rate"],
            "std_click_rate": agg["std_click_rate"],
            "mean_alignment": agg["mean_alignment"],
            "std_alignment": agg["std_alignment"],
            "realism": realism.get(tag),
        }

    # Baseline = first un-steered approach if present, else the first listed.
    baseline_tag = next(
        (t for t in ordered if not found[t].get("steer", True)), ordered[0]
    )
    base = rows[baseline_tag]

    has_realism = any(r["realism"] for r in rows.values())
    realism_method = next(
        (r["realism"]["method"] for r in rows.values() if r["realism"]), None
    )

    lines = [
        "=" * 88,
        "APPROACH COMPARISON  (baseline vs steered)",
        "=" * 88,
        f"Baseline = {base['display']}",
        "",
        f"{'Approach':<40}{'Click rate':>16}{'Alignment':>16}"
        + (f"{'Realism (FID)':>16}" if has_realism else ""),
        "-" * 88,
    ]
    for tag in ordered:
        r = rows[tag]
        line = (
            f"{r['display']:<40}"
            f"{100*r['mean_click_rate']:>9.1f}% (±{100*r['std_click_rate']:>4.1f})"
            f"{r['mean_alignment']:>10.3f} (±{r['std_alignment']:>5.3f})"
        )
        if has_realism:
            rv = r["realism"]
            line += f"{rv['score']:>16.2f}" if rv else f"{'n/a':>16}"
        lines.append(line)

    lines += ["-" * 88, "", f"Delta vs baseline ({base['display']}):"]
    for tag in ordered:
        if tag == baseline_tag:
            continue
        r = rows[tag]
        d_click = 100 * (r["mean_click_rate"] - base["mean_click_rate"])
        d_align = r["mean_alignment"] - base["mean_alignment"]
        seg = (
            f"  {r['display']:<40} "
            f"click {d_click:+.1f} pts   alignment {d_align:+.3f}"
        )
        if has_realism and r["realism"] and base["realism"]:
            seg += f"   realism {r['realism']['score'] - base['realism']['score']:+.2f}"
        lines.append(seg)
    if has_realism:
        lines += ["", f"(realism = InceptionV3 {realism_method}; lower = closer to real shoes)"]
    lines += ["", "=" * 88]

    report_text = "\n".join(lines)
    print(report_text)
    (output_path / "approach_comparison.txt").write_text(report_text, encoding="utf-8")

    comparison = {
        "baseline": baseline_tag,
        "modes": rows,
        "realism_method": realism_method,
    }
    (output_path / "approach_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    # Grouped bars: click rate, alignment, and realism (if available).
    panels = [
        ("mean_click_rate", "std_click_rate", "Mean Click Rate (%)", 100.0),
        ("mean_alignment", "std_alignment", "Mean Alignment (cosine)", 1.0),
    ]
    fig, axes = plt.subplots(1, len(panels) + (1 if has_realism else 0), figsize=(6 * (len(panels) + (1 if has_realism else 0)), 5))
    if not hasattr(axes, "__len__"):
        axes = [axes]
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(ordered), 1)))
    bar_colors = [palette[i] for i in range(len(ordered))]
    x = np.arange(len(ordered))
    labels = [rows[t]["display"] for t in ordered]

    for ax, (mkey, skey, ylabel, scale) in zip(axes, panels):
        means = [rows[t][mkey] * scale for t in ordered]
        errs = [rows[t][skey] * scale for t in ordered]
        ax.bar(x, means, yerr=errs, capsize=8, color=bar_colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" (")[0])
        ax.grid(True, axis="y", alpha=0.3)

    if has_realism:
        ax = axes[len(panels)]
        scores = [rows[t]["realism"]["score"] if rows[t]["realism"] else 0 for t in ordered]
        ax.bar(x, scores, color=bar_colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(f"Realism ({realism_method}, lower=better)")
        ax.set_title("Image Realism vs Real Shoes")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = output_path / "approach_comparison.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")
    plt.close()

    return comparison


if __name__ == "__main__":
    # Re-run ONLY the eval comparison from already-saved results -- no training,
    # no diffusion generation. Reads test_results_*.json (and the saved
    # generated_images/ for the realism metric) from --results-dir.
    import argparse
    import sys

    # Repo root on sys.path so the reused InceptionV3 realism metric
    # (evals.compute_fid_metrics) imports when run as a script.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser(
        description="Rebuild the eval comparison from saved results (no re-run)."
    )
    ap.add_argument("--results-dir", default="preference/results")
    ap.add_argument(
        "--real-image-dir",
        default="sneakerdata/images/shoes_100",
        help="Real images for the InceptionV3 realism metric; pass '' to skip it",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    # Load saved results by their JSON `mode` label (LoRA fixed/adaptive use the
    # unsuffixed names) to regenerate the two-mode convergence/alignment plots.
    def _load(label):
        f = results_dir / f"test_results_{label}.json"
        return json.loads(f.read_text()) if f.exists() else None

    res_fixed = _load("fixed_theta")
    res_adaptive = _load("adaptive_theta")
    if res_fixed and res_adaptive:
        plot_convergence_curves(res_fixed, res_adaptive, str(results_dir))
        plot_preference_alignment(res_fixed, res_adaptive, str(results_dir))
        generate_experiment_report(res_fixed, res_adaptive, str(results_dir))

    # Compare every solution found in the results dir (scans all
    # test_results_*.json, including base-SD1.5 and other variants).
    result = compare_approaches(
        output_dir=str(results_dir),
        real_image_dir=(args.real_image_dir or None),
    )
    if result is None:
        raise SystemExit(
            f"Need >=2 test_results_*.json in {results_dir} to compare; "
            "run the experiment first."
        )
