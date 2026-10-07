# Preference Steering: Real-Time Personalized Generation

A complete system for learning universal human preferences and using them to generate personalized product images via diffusion model steering.

---

## 🚀 Quick Start (5 minutes)

**Prerequisites:**
```bash
pip install -r preference/requirements.txt
```

**Full pipeline — three stages, in order.** Generation needs a sneaker LoRA, so
the LoRA is trained first, then the shared θ, then evaluation with real steered
generation:

```bash
# === STAGE 1: train the sneaker LoRA (SD1.5 @ 512px, "a sneaker" prompt) ===
python preference/training/train_sneaker_lora.py \
  --image-dir sneakerdata/images/shoes_100 \
  --output preference/models_checkpoints/sneaker_lora \
  --epochs 30
# Quick sanity check first:
#   --epochs 5 --max-train-images 20 --num-samples 4
# then eyeball preference/models_checkpoints/sneaker_lora/samples/ before continuing.

# === STAGE 2 + 3: train shared θ, then evaluate with steered generation ===
# Smoke test first (1 user, 3 rounds, 15 steps) — confirms real images generate:
python preference/experiments/full_preference_experiment.py \
  --smoke --lora preference/models_checkpoints/sneaker_lora

# Full run:
python preference/experiments/full_preference_experiment.py \
  --lora preference/models_checkpoints/sneaker_lora
```

Data prep (download + CLIP labeling) runs **automatically** inside the experiment;
the driver re-labels whenever the feature caches are missing or stale. To run it
standalone: `python preference/data_prep/download_zappos.py`.

**Experiment CLI flags** (`full_preference_experiment.py`):
- `--lora PATH` — trained sneaker LoRA dir from stage 1. Omit to use base SD1.5 (generic sneakers) with a warning.
- `--smoke` — tiny validation run: 1 user, 3 rounds, 15 inference steps, per-step steering log on.
- `--no-real-diffusion` — skip the diffusion model and use the fast feature-space proxy (no images).
- `--prompt "..."` — generation prompt (default `"a sneaker"`).
- plus `--num-train-users`, `--interactions-per-user`, `--num-test-users`, `--rounds-per-test-user`, `--theta-lr`, `--user-lr`, `--results-dir`.

**Stage 1 CLI flags** (`train_sneaker_lora.py`):
- `--epochs`, `--batch-size`, `--lr`, `--image-size` (default 512), `--model-id` (default SD1.5).
- `--max-train-images N` — train on only the first N images (quick sanity check).
- `--num-samples N` — sample images dumped to `<output>/samples/` after training (0 to skip).

**Outputs:**
- `preference/models_checkpoints/sneaker_lora/` — trained LoRA + `samples/` preview images (stage 1)
- `preference/results/generated_images/user{ID}_{fixed|adaptive}_round{NNN}.png` — every steered image (stage 3)
- `preference/results/test_results_fixed_theta.json` / `test_results_adaptive_theta.json` — per-round metrics
- `preference/results/convergence_comparison.png` — click rate over time
- `preference/results/preference_alignment.png` — how well learned preferences match ground truth
- `preference/results/experiment_report.txt` — summary statistics

**For RunPod:** Clone repo, run `pip install -r preference/requirements.txt`, then execute the three stages above in order.

> **Diffusion import error?** If `import diffusers` crashes with
> `infer_schema(func): Parameter q has unsupported type torch.Tensor`
> (`attention_dispatch.py` / `_flash_attn_3`), the pod has a too-new diffusers.
> The pipeline is tested on **`diffusers==0.30.0`** (pinned in requirements);
> fix in place with `pip install "diffusers==0.30.0"`. diffusers ≥ 0.32 registers
> a Flash-Attention-3 custom op at import time that needs torch ≥ 2.6.

---

## Saving & reusing models (Hugging Face)

Both trained artifacts — the sneaker **LoRA** and the **preference model (theta)** —
can be exported into one self-contained bundle and pushed to Hugging Face, so a
future pod pulls instead of retraining.

**Push only — no training/eval/diffusion** (assembles the bundle from artifacts
already on disk and syncs):
```bash
python preference/utils/hf_artifacts.py \
  --hf-target hf://buckets/tkher/preferencemodels
# defaults: --lora preference/models_checkpoints/sneaker_lora
#           --theta preference/results/training_metadata.json
#           --concept-stats sneakerdata/features/concept_stats.json
#           --out preference/exports/preferencemodels
# omit --hf-target to just build the bundle and print the `hf upload` command.
```
The `--hf-target` is normalized to an `owner/name` repo id, so
`tkher/preferencemodels`, `hf://buckets/tkher/preferencemodels`, or the full
`https://huggingface.co/...` URL all resolve to `tkher/preferencemodels`.
Auth uses ambient creds (`hf auth login` / `HF_TOKEN`) or pass `--hf-token`.

**Export during a run** (add to the Stage 2+3 command):
```bash
python preference/experiments/full_preference_experiment.py \
  --lora preference/models_checkpoints/sneaker_lora \
  --export-dir preference/exports/preferencemodels \
  --hf-sync-target hf://buckets/tkher/preferencemodels   # optional: uploads after the run
```
`--export-dir` writes the bundle; `--hf-sync-target` additionally uploads it.
You can also upload the folder yourself:
```bash
hf upload tkher/preferencemodels ./preference/exports/preferencemodels . --repo-type model
```

**Bundle folder structure** (what gets synced / pulled):
```
preferencemodels/
├── lora/                          # sneaker LoRA (diffusers format)
│   └── pytorch_lora_weights.safetensors
├── preference_model/
│   ├── theta.json                 # trained shared preference vector
│   └── concept_stats.json         # CLIP feature-transform stats (mean/std/temp)
├── eval_metrics/                  # run metrics: test_results_*.json, plots,
│                                  #   experiment_report.txt, metrics.sqlite
├── manifest.json                  # base model, concepts, image size, prompt, date
└── README.md
```
Auth: an explicit `--hf-token` wins; otherwise ambient creds (`hf auth login` /
`HF_TOKEN`) are used; if none exist you'll be prompted for a token. The upload is
always created `--private`.

**Pull & reuse next time** (skips LoRA + theta training):
```bash
# 1. pull the bundle back
hf download tkher/preferencemodels --repo-type model \
  --local-dir ./preference/exports/preferencemodels
# 2. run eval straight from it (theta loaded, LoRA loaded, no retraining)
python preference/experiments/full_preference_experiment.py \
  --load-bundle preference/exports/preferencemodels --smoke
```
`--load-bundle` loads `theta.json` into the oracle and points generation at the
bundle's `lora/`, so only data prep + evaluation run.

---

## Metrics database (every example + every step)

Beyond the summary JSON/plots, every run writes a **durable SQLite DB** at
`<results_dir>/metrics.sqlite`, committing a row **as each example is produced**
(crash-resilient over multi-hour runs). On by default; disable with
`config.log_metrics_db = False`.

**Schema** (`preference/utils/metrics_db.py`):

| table | grain | key columns |
|---|---|---|
| `runs` | one per invocation | `run_id`, `created_utc`, `base_model`, `lora`, `config_json`, `git_commit` |
| `examples` | one per user × round × mode | `feature_space`, `mode`, `phase` (bootstrap/generation), `user_id`, `round`, `clicked`, `pref_score`, `theta_dot_f`, `uA_dot_f`, `alignment_uA_true`, `cum_click_rate`, `image_path`, `f_json`, `uA_json`, `theta_json`, `true_pref_json` |
| `steps` | one per steered denoising step | `user_id`, `round`, `step_idx`, `timestep`, `pref_score`, `grad_norm`, `steered` |

Vectors are JSON text, so the same schema covers 3-d concepts and 512-d embeddings.

**Aggregate / visualize later:**
```python
import sqlite3, pandas as pd
con = sqlite3.connect("preference/results/metrics.sqlite")
ex = pd.read_sql("SELECT * FROM examples", con)

# convergence: click rate vs round, per config
ex[ex.phase == "generation"].groupby(["feature_space", "mode", "round"]).clicked.mean()

# preference alignment trajectory
ex.groupby(["mode", "round"]).alignment_uA_true.mean()

# within-image steering trajectory
steps = pd.read_sql("SELECT * FROM steps", con)
steps.groupby("step_idx").pref_score.mean()
```

---

## Overview

**Problem:** How do we generate personalized product images in real-time without building a separate model per user?

**Solution:** 
1. Learn a **shared preference model (θ)** from many users' interactions
2. Represent each user efficiently as a **small per-user vector (u_A)**  
3. At generation time, use inference-time diffusion steering to create images tailored to (θ, u_A)

**Key insight:** User preferences follow transferable patterns. Learn the patterns once, then personalize via a tiny per-user vector.

---

## Architecture

```
SHARED (learned offline):
  θ ∈ ℝ³         — universal human preference model
  (K=3 feature dimensions: colorful, chunky, clean)

PER-USER (efficient):
  u_A ∈ ℝ³       — user A's preference weights (learned from clicks)

AT GENERATION:
  preference_score = (θ + u_A) · features(image)
  ∇ score steers diffusion latents toward high-preference images
```

---

## Project Structure

```
preference/
├── data_prep/                    # Data preparation pipeline
│   ├── download_zappos.py        # Download 100 Zappos shoe images
│   ├── label_features_with_ai.py # AI-label images with 3 features
│   └── __init__.py
│
├── models/                       # Core model components
│   ├── preference_oracle.py      # PreferenceOracle: scores images via θ·f(x) + u_A·f(x)
│   ├── feature_extractor.py      # DifferentiableFeatureExtractor: CLIP-based
│   └── __init__.py
│
├── inference/                    # Inference-time steering
│   ├── preference_oracle_inference.py  # Wrapper for differentiable scoring
│   ├── preference_steering.py          # PreferenceSteeredGenerator (main loop)
│   └── __init__.py
│
├── training/                     # Training phase
│   ├── train_sneaker_lora.py     # Stage 1: SD1.5 LoRA on the 100 sneaker images
│   └── __init__.py               # (preference simulators live in simulation/)
│
├── simulation/                   # Simulated user experiments
│   ├── synthetic_user_simulator.py     # SyntheticUser class
│   ├── simulate_training_phase.py      # Learn θ from 50 users
│   ├── simulate_test_phase.py          # Test on held-out users
│   └── __init__.py
│
├── evals/                        # Evaluation and visualization
│   ├── evaluate_preference_steering.py # Metrics, plots, reports
│   └── __init__.py
│
├── experiments/                  # Orchestration
│   ├── config.py                 # PreferenceExperimentConfig (all hyperparams)
│   ├── full_preference_experiment.py   # Main entry point
│   └── __init__.py
│
└── results/                      # Output directory (created at runtime)
    ├── test_results_fixed_theta.json
    ├── test_results_adaptive_theta.json
    ├── convergence_comparison.png
    ├── preference_alignment.png
    └── experiment_report.txt

sneakerdata/                      # Data (independent from existing /data/)
├── images/
│   └── shoes_100/                # 100 downloaded Zappos sneaker images
├── features/
│   └── feature_labels.json       # Pre-extracted CLIP feature vectors
└── splits/                       # Train/val/test splits (if needed)
```

---

## Quick Start

### 1. Run Full Experiment

```bash
cd /c/Users/tlocal/Desktop/vlm-binding
python preference/experiments/full_preference_experiment.py \
  --lora preference/models_checkpoints/sneaker_lora \
  --num-train-users 50 \
  --interactions-per-user 100 \
  --num-test-users 3 \
  --rounds-per-test-user 50
```

This runs:
- **Data Prep:** Download 100 Zappos images, label features, validate
- **Training:** Learn shared θ from 50 synthetic users
- **Test (Fixed):** Generate for held-out users, θ frozen
- **Test (Adaptive):** Generate for held-out users, θ updates
- **Eval:** Plot convergence, alignment, generate report

### 2. Run Individual Steps

**Download data:**
```bash
python preference/data_prep/download_zappos.py
```

**Label features:**
```bash
python preference/data_prep/label_features_with_ai.py
```

**Validate feature space:**
```bash
python preference/models/feature_extractor.py
```

**Train θ:**
```bash
python preference/simulation/simulate_training_phase.py
```

**Test (with custom oracle):**
```python
from preference.simulation.simulate_test_phase import simulate_test_phase
results = simulate_test_phase(
    trained_oracle=my_oracle,
    adapt_theta=True,  # or False for fixed mode
)
```

---

## Key Components

### PreferenceOracle
```python
from preference.models.preference_oracle import PreferenceOracle

oracle = PreferenceOracle(num_features=3)  # 3D preference space
score = oracle(features, user_preferences)  # differentiable
oracle.update_theta(gradient, learning_rate=0.01)
```

### DifferentiableFeatureExtractor
```python
from preference.models.feature_extractor import DifferentiableFeatureExtractor

extractor = DifferentiableFeatureExtractor(
    feature_concepts=[
        "colorful bright sneaker",
        "chunky platform sneaker",
        "clean minimal sneaker",
    ]
)

# Differentiable CLIP-based feature extraction
features = extractor.extract_features_from_tensor(image_tensor)  # [B, 3]
```

### PreferenceSteeredGenerator
```python
from preference.inference.preference_steering import PreferenceSteeredGenerator
from preference.inference.preference_oracle_inference import PreferenceOracleInference

# Initialize
preference_oracle_inf = PreferenceOracleInference(oracle, extractor)
generator = PreferenceSteeredGenerator(pipeline, preference_oracle_inf)

# Generate
config = PreferenceSteeringConfig(steering_strength=1.0, num_inference_steps=30)
image = generator.generate(
    prompt="a sneaker",
    config=config,
    user_preferences=user_pref_vector,  # [3]
)
```

### SyntheticUser
```python
from preference.simulation.synthetic_user_simulator import SyntheticUser

user = SyntheticUser(
    user_id=0,
    true_preference=torch.tensor([0.8, 0.1, 0.1]),  # loves colorful
)

clicked = user.decide_click(features)  # bool
```

---

## Configuration

All hyperparameters in `preference/experiments/config.py`:

```python
config = PreferenceExperimentConfig(
    # Data
    num_zappos_images=100,
    feature_concepts=["colorful...", "chunky...", "clean..."],
    
    # Training
    num_synthetic_train_users=50,
    interactions_per_train_user=100,
    theta_learning_rate=0.01,
    
    # Testing
    num_synthetic_test_users=3,
    rounds_per_test_user=50,
    user_learning_rate=0.05,
    adapt_theta=False,  # Toggle: fixed vs. adaptive
    
    # Steering
    steering_strength=1.0,
    guidance_scale=7.5,
    num_inference_steps=30,
)
```

---

## Experiment Modes

### Fixed Theta (Baseline)
- θ learned offline, frozen during test
- Only u_A updates from test user interactions
- **Tests:** Does θ generalize to unseen users?
- **Expected:** Moderate baseline performance

### Adaptive Theta (Real-World)
- θ learned offline, continues to update during test
- Both θ AND u_A evolve from test user feedback
- **Tests:** How fast does the model adapt to emerging trends?
- **Expected:** Better performance over time; realistic scenario

---

## Key Outputs

After running `full_preference_experiment.py`:

1. **`test_results_fixed_theta.json`**
   - Per-user click rates, preference alignment scores
   - Round-by-round detailed logs

2. **`test_results_adaptive_theta.json`**
   - Same metrics but with θ adapting in real-time

3. **`convergence_comparison.png`**
   - Left: Individual user click trajectories (moving average)
   - Right: Aggregate bar chart comparing modes

4. **`preference_alignment.png`**
   - How well each user's learned u_A matches their true preference

5. **`experiment_report.txt`**
   - Summary statistics: mean click rate, alignment, delta between modes

---

## Mathematical Foundation

**Per-step steering loss:**
```
L_pref = -score
       = -(θ + u_A) · f(x̂₀)
```

**Gradient path (differentiable):**
```
z_t → x̂₀ (DDIM) → image → CLIP encoder → f(x̂₀) → score → L → ∇_{z_t} L
```

**Updates:**
```
User clicked:    u_A ← u_A + α·f(x)    θ ← θ + β·f(x)  (if adaptive)
User skipped:    u_A ← u_A - α·f(x)    θ ← θ - β·f(x)  (if adaptive)
```

**Evaluation metrics:**
- **Click rate:** % of generated images user clicks (reflects preference alignment)
- **Alignment:** cos_sim(u_A, true_preference) (how well learned = ground truth)
- **Cumulative regret:** Standard bandit metric (not computed here, but could be)

---

## Independence from Existing System

✅ **Completely separate:**
- Data: `/sneakerdata/` (vs. existing `/data/`)
- Code: `/preference/` (vs. existing `/inference/`, `/models/`, etc.)
- No modifications to existing files
- Can run in parallel with cost-steering system

❌ **Does NOT reuse:**
- Cost oracle from jewelry system
- Jewelry dataset or LoRA

✅ **CAN reuse (if desired):**
- `DiffusionGeneratorLoRA` pipeline wrapper (currently assumes SD1.5/SDXL, compatible)
- General training/eval patterns (but preference system is self-contained)

---

## Next Steps for Full Integration

1. ✅ **Real diffusion steering (done):**
   - `simulate_test_phase.py` now calls the real `PreferenceSteeredGenerator.generate()` when a generator is supplied (default when `--lora` is set); the feature-space proxy remains available via `--no-real-diffusion`.
   - Sneaker LoRA is trained in stage 1 via `preference/training/train_sneaker_lora.py`.

2. **Train real preference models:**
   - Instead of synthetic users, collect real user click data
   - Update θ from real preferences instead of simulation

3. **Deploy:** 
   - Use trained θ + efficient u_A update to personalize generation in production
   - Stream personalization: show image → collect feedback → update u_A → generate next

---

## Debugging / Development

**Validate feature space:**
```bash
python preference/models/feature_extractor.py
```
Shows: sample images and their CLIP feature scores. Verify that labeled features correlate with CLIP scores.

**Check training phase:**
```bash
python preference/simulation/simulate_training_phase.py
```
Shows: theta convergence, click statistics, final learned theta.

**Manual test:**
```python
from preference.experiments.config import PreferenceExperimentConfig
from preference.simulation.simulate_training_phase import simulate_training_phase

config = PreferenceExperimentConfig(
    num_synthetic_train_users=5,  # tiny for debug
    interactions_per_train_user=20,
)
oracle, _ = simulate_training_phase(
    num_synthetic_users=5,
    interactions_per_user=20,
)
print("Learned theta:", oracle.get_theta())
```

---

## Authors & Citation

Developed as the preference-steered component of SteeringDiffusion.

**Paper:** "Personalized Product Generation via Real-Time Preference Learning" (POMS Journal)

---
