"""Export / load / sync a self-contained preference bundle.

A bundle gathers everything needed to *reuse* a trained run without retraining:

    <bundle>/
    ├── lora/                          # sneaker LoRA (diffusers format)
    │   └── pytorch_lora_weights.safetensors
    ├── preference_model/
    │   ├── theta.json                 # trained shared preference vector
    │   └── concept_stats.json         # CLIP feature-transform stats (mean/std/temp)
    ├── eval_metrics/                  # run metrics (results json, plots, report, sqlite)
    ├── manifest.json                  # base model, concepts, image size, prompt, date
    └── README.md                      # how to load it back

Push it to Hugging Face with the `hf` CLI, e.g.:

    hf sync ./<bundle> hf://buckets/tkher/preferencemodels

and pull it back the same way (swap the arguments).
"""

from __future__ import annotations

import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Allow running this file directly as a script (python preference/utils/hf_artifacts.py).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preference.models.preference_oracle import PreferenceOracle


# Everything under the results dir is copied into <bundle>/eval_metrics/ EXCEPT
# these names (matched at any depth) -- generated_images/ is excluded to keep the
# upload small; the rest of results/ (every json/png/txt/sqlite/report) goes up.
_EVAL_METRICS_EXCLUDE = ("generated_images",)


def export_preference_bundle(
    export_dir: str,
    lora_dir: str | None,
    oracle: PreferenceOracle,
    concept_stats_file: str | None,
    manifest: dict | None = None,
    results_dir: str | None = None,
) -> str:
    """Assemble a structured bundle of LoRA + preference model + eval metrics."""
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. LoRA weights
    if lora_dir and Path(lora_dir).exists():
        lora_out = out / "lora"
        if lora_out.exists():
            shutil.rmtree(lora_out)
        shutil.copytree(lora_dir, lora_out)
        lora_rel = "lora/"
    else:
        print(f"[export] WARNING: lora_dir {lora_dir!r} missing; bundle has no LoRA")
        lora_rel = None

    # 2. Preference model (theta) + feature-transform stats
    pm = out / "preference_model"
    pm.mkdir(parents=True, exist_ok=True)
    oracle.save(str(pm / "theta.json"))
    stats_rel = None
    if concept_stats_file and Path(concept_stats_file).exists():
        shutil.copy(concept_stats_file, pm / "concept_stats.json")
        stats_rel = "preference_model/concept_stats.json"

    # 3. Eval metrics: the entire results dir EXCEPT generated_images/.
    eval_rel = None
    if results_dir and Path(results_dir).exists():
        res = Path(results_dir)
        em = out / "eval_metrics"
        if em.exists():
            shutil.rmtree(em)
        shutil.copytree(
            res, em, ignore=shutil.ignore_patterns(*_EVAL_METRICS_EXCLUDE)
        )
        copied = sorted(p.name for p in em.rglob("*") if p.is_file())
        if copied:
            eval_rel = "eval_metrics/"
            print(f"[export] results copied to eval_metrics/ ({len(copied)} files, "
                  f"excluding {', '.join(_EVAL_METRICS_EXCLUDE)})")
        else:
            shutil.rmtree(em)
            print(f"[export] WARNING: no eval metrics found in {results_dir}")
    elif results_dir:
        print(f"[export] WARNING: results_dir {results_dir!r} missing; no eval metrics")

    # 4. Manifest + README
    manifest = dict(manifest or {})
    manifest.update(
        {
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "layout": {
                "lora": lora_rel,
                "theta": "preference_model/theta.json",
                "concept_stats": stats_rel,
                "eval_metrics": eval_rel,
            },
        }
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "README.md").write_text(_render_readme(manifest), encoding="utf-8")

    print(f"[export] Preference bundle written to {out}")
    return str(out)


def load_preference_bundle(bundle_dir: str, device: str = "cuda") -> dict:
    """Load a bundle written by `export_preference_bundle`.

    Returns dict with keys: lora_dir, oracle, concept_stats, manifest.
    """
    b = Path(bundle_dir)
    manifest_path = b / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    lora_dir = b / "lora"
    theta_path = b / "preference_model" / "theta.json"
    concept_stats = b / "preference_model" / "concept_stats.json"

    oracle = (
        PreferenceOracle.load(str(theta_path), device=device)
        if theta_path.exists()
        else None
    )
    return {
        "lora_dir": str(lora_dir) if lora_dir.exists() else None,
        "oracle": oracle,
        "concept_stats": str(concept_stats) if concept_stats.exists() else None,
        "manifest": manifest,
    }


def _parse_repo_id(target: str) -> str:
    """Normalize a target into a Hugging Face repo id (owner/name).

    Accepts forms like:
        tkher/preferencemodels
        hf://buckets/tkher/preferencemodels
        https://huggingface.co/buckets/tkher/preferencemodels
    A leading scheme/host and a leading `buckets/` segment (not a real HF path)
    are stripped, leaving owner/name.
    """
    t = target.strip()
    for prefix in (
        "https://huggingface.co/",
        "http://huggingface.co/",
        "hf://",
    ):
        if t.startswith(prefix):
            t = t[len(prefix):]
    t = t.strip("/")
    parts = [p for p in t.split("/") if p]
    if len(parts) >= 3 and parts[0] in ("buckets", "models", "model", "datasets"):
        parts = parts[1:]
    if len(parts) < 2:
        raise SystemExit(
            f"Could not derive an owner/name repo id from {target!r}. "
            "Pass a repo id like 'tkher/preferencemodels'."
        )
    return "/".join(parts[:2])


def push_to_hf(local_dir: str, target: str, token: str | None = None) -> None:
    """Upload a bundle folder to a private Hugging Face model repo via `hf upload`.

    `hf sync` does not exist in the Hub CLI; the push verb is `hf upload`.
    Auth: an explicit `token` wins; otherwise ambient credentials are used
    (`hf auth login` / HF_TOKEN). If neither is present, prompt for a token so
    the upload can't silently fail on auth.
    """
    repo_id = _parse_repo_id(target)
    print(f"[export] Resolved HF repo: {repo_id}  (from {target})")

    if not token:
        try:
            from huggingface_hub import get_token

            ambient = get_token()
        except Exception:
            ambient = None
        if not ambient:
            import getpass

            token = getpass.getpass("Enter your Hugging Face token: ").strip() or None

    cmd = ["hf", "upload", repo_id, str(local_dir), ".", "--repo-type", "model", "--private"]
    if token:
        cmd += ["--token", token]
    print(f"[export] Running: hf upload {repo_id} {local_dir} . --repo-type model --private")
    subprocess.run(cmd, check=True)


def _oracle_from_theta_file(theta_path: str) -> PreferenceOracle:
    """Build a PreferenceOracle from a theta.json OR a training_metadata.json."""
    import torch

    with open(theta_path) as f:
        data = json.load(f)
    if "theta" in data:  # theta.json (from a previous bundle export)
        theta, num_features = data["theta"], data.get("num_features", len(data["theta"]))
    elif "final_theta" in data:  # training_metadata.json (from a training run)
        theta = data["final_theta"]
        num_features = len(theta)
    else:
        raise SystemExit(
            f"{theta_path} has neither 'theta' nor 'final_theta'. "
            "Point --theta at preference/results/training_metadata.json or a theta.json."
        )
    oracle = PreferenceOracle(num_features=num_features, device="cpu")
    oracle.set_theta(torch.tensor(theta, dtype=torch.float32))
    return oracle


def _render_readme(manifest: dict) -> str:
    return (
        "# Preference bundle\n\n"
        "Self-contained artifacts for the preference-steering pipeline.\n\n"
        "## Contents\n"
        "- `lora/` — sneaker LoRA (diffusers format)\n"
        "- `preference_model/theta.json` — trained shared preference vector (theta)\n"
        "- `preference_model/concept_stats.json` — CLIP feature-transform stats\n"
        "- `eval_metrics/` — run metrics (results json, plots, report, metrics.sqlite)\n\n"
        "## Reuse (skip retraining)\n"
        "```\n"
        "python preference/experiments/full_preference_experiment.py \\\n"
        "  --load-bundle <this_dir>\n"
        "```\n\n"
        "## Manifest\n"
        "```json\n" + json.dumps(manifest, indent=2) + "\n```\n"
    )


def main() -> None:
    """Assemble a bundle from already-trained artifacts and (optionally) push it.

    No training, no diffusion, no evaluation -- just gather files and sync.

        python preference/utils/hf_artifacts.py \\
          --hf-target hf://buckets/tkher/preferencemodels
    """
    import argparse

    ap = argparse.ArgumentParser(
        description="Push trained LoRA + preference model to Hugging Face (no run)."
    )
    ap.add_argument(
        "--lora",
        default="preference/models_checkpoints/sneaker_lora",
        help="Trained sneaker LoRA directory",
    )
    ap.add_argument(
        "--theta",
        default="preference/results/training_metadata.json",
        help="theta.json or training_metadata.json holding the trained preference model",
    )
    ap.add_argument(
        "--concept-stats",
        default="sneakerdata/features/concept_stats.json",
        help="CLIP feature-transform stats",
    )
    ap.add_argument(
        "--out",
        default="preference/exports/preferencemodels",
        help="Local bundle directory to assemble",
    )
    ap.add_argument(
        "--results-dir",
        default="preference/results",
        help="Run results dir; its eval metrics are copied into eval_metrics/",
    )
    ap.add_argument(
        "--hf-target",
        default=None,
        help="If set, upload the bundle to this HF repo "
        "(repo id like tkher/preferencemodels, or an hf://.../https URL)",
    )
    ap.add_argument(
        "--hf-token",
        default=None,
        help="HF token (otherwise uses ambient `hf auth login` / HF_TOKEN)",
    )
    ap.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--prompt", default="a sneaker")
    args = ap.parse_args()

    oracle = _oracle_from_theta_file(args.theta)
    manifest = {
        "base_model": args.base_model,
        "generation_prompt": args.prompt,
        "num_features": oracle.num_features,
        "source_theta": args.theta,
    }
    out = export_preference_bundle(
        export_dir=args.out,
        lora_dir=args.lora,
        oracle=oracle,
        concept_stats_file=args.concept_stats,
        manifest=manifest,
        results_dir=args.results_dir,
    )
    if args.hf_target:
        push_to_hf(out, args.hf_target, token=args.hf_token)
    else:
        print(
            f"\nBundle ready at {out}\nPush it with:\n"
            f"  hf upload tkher/preferencemodels {out} . --repo-type model"
        )


if __name__ == "__main__":
    main()
