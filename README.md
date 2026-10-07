# Aligning Text-to-2D Diffusion Models with Manufacturing Cost Constraints using Differentiable Rewards

This project supports:
1. Clean oracle training (price prediction from clean images)
2. Diffusion generation with optional inference-time cost steering
3. Local or Hugging Face model loading workflows

## Data Format

Expected CSV columns:
- `image_path` (or `id` if you map IDs to image files)
- `name`
- `elements`
- `why_love`
- `dimensions`
- `price`

One text prompt is built per row from the text fields.

Default dataset file:
- `data/catalog_clip_ready.csv`

Default image root:
- `data/images`

## Environment

Activate virtual environment (Git Bash on Windows):

```bash
source /c/Users/tlocal/Desktop/vlm-binding/.venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Standard Commands

### End-to-end command order (recommended)

1. Train clean oracle
2. Train diffusion LoRA
3. (Optional) Push diffusion + oracle to Hugging Face
4. Run inference
5. Run budget-delta evaluation (with saved images)

### 1) Retrain both oracle and diffusion locally (bracelets only)

Oracle:

```bash
python training/train_oracle.py \
	--csv data/catalog_clip_ready.csv \
	--image-root data/images \
	--output models/oracle \
	--epochs 20 \
	--batch-size 16 \
	--bracelets-only
```

Diffusion LoRA:

```bash
python training/train_diffusion_lora.py \
	--csv data/catalog_clip_ready.csv \
	--output models/diffusion_lora \
	--epochs 5 \
	--batch-size 4 \
	--bracelets-only
```

### 2) Use diffusion model from Hugging Face + train oracle locally

Train oracle (bracelets only):

```bash
python training/train_oracle.py \
	--csv data/catalog_clip_ready.csv \
	--image-root data/images \
	--output models/oracle \
	--epochs 20 \
	--batch-size 16 \
	--bracelets-only
```

Run inference using your HF diffusion model with steering:

```bash
python inference/generate_designs.py \
	--mode batch \
	--prompts "silver bracelet with blue stones" \
	--diffusion-source hf \
	--hf-diffusion-model-id <your-hf-diffusion-model> \
	--oracle-checkpoint models/oracle/checkpoints/best.pt \
	--steering \
	--target-price 120 \
	--num-images 4 \
	--steps 20
```

### 3) Pull both diffusion and oracle from Hugging Face

Interactive (prompts for token and repo IDs):

```bash
python utils/pull_remote_weights.py
```

Non-interactive:

```bash
python utils/pull_remote_weights.py \
	--hf-token <hf_token> \
	--diffusion-repo <user/diffusion_repo> \
	--oracle-repo <user/oracle_repo> \
	--diffusion-local-dir models/diffusion_hf \
	--oracle-local-dir models/oracle_hf
```

Then run inference from local pulled artifacts by pointing to your model path/setup.

### 4) Train diffusion from scratch locally, then push diffusion + oracle to one HF repo

Train oracle first:

```bash
python training/train_oracle.py \
	--csv data/catalog_clip_ready.csv \
	--image-root data/images \
	--output models/oracle \
	--epochs 20 \
	--batch-size 16 \
	--bracelets-only
```

Train diffusion LoRA:

```bash
python training/train_diffusion_lora.py \
	--csv data/catalog_clip_ready.csv \
	--output models/diffusion_lora \
	--epochs 5 \
	--batch-size 4 \
	--bracelets-only
```

Export a full diffusion pipeline (base + fused LoRA), bundle oracle weights, and upload to one HF model repo:

```bash
python utils/push_remote_weights.py \
	--repo-id <user/repo_name> \
	--base-model-id stabilityai/stable-diffusion-xl-base-1.0 \
	--lora-root models/diffusion_lora \
	--oracle-checkpoint models/oracle/checkpoints/best.pt \
	--oracle-dir models/oracle \
	--private
```

This uploads:
- diffusion model at repo root (directly loadable via `from_pretrained(<user/repo_name>)`)
- oracle at `oracle/best.pt` in the same repo
- `upload_manifest.json` with provenance details

### 5) Run inference after training

Local diffusion (LoRA path):

```bash
python inference/generate_designs.py \
	--mode batch \
	--prompts "silver bracelet with blue stones" "minimal gold chain bracelet" \
	--diffusion-source local-finetune \
	--lora-dir models/diffusion_lora \
	--oracle-checkpoint models/oracle/checkpoints/best.pt \
	--steering \
	--target-price 120 \
	--num-images 4 \
	--steps 20
```

HF diffusion repo:

```bash
python inference/generate_designs.py \
	--mode batch \
	--prompts "silver bracelet with blue stones" \
	--diffusion-source hf \
	--hf-diffusion-model-id <user/repo_name> \
	--oracle-checkpoint models/oracle/checkpoints/best.pt \
	--steering \
	--target-price 120 \
	--num-images 4 \
	--steps 20
```

## Inference Modes

### Interactive generation

```bash
python inference/generate_designs.py \
	--mode interactive \
	--diffusion-source local-finetune \
	--lora-dir models/diffusion_lora \
	--oracle-checkpoint models/oracle/checkpoints/best.pt \
	--steering
```

### Batch generation

```bash
python inference/generate_designs.py \
	--mode batch \
	--prompts "silver bracelet with blue stones" "gold heart bracelet" \
	--diffusion-source local-finetune \
	--lora-dir models/diffusion_lora \
	--oracle-checkpoint models/oracle/checkpoints/best.pt \
	--steering \
	--target-price 120 \
	--num-images 4 \
	--steps 20
```

## Budget-Error Delta Experiment (A/B)

Compares no-steer vs steer and prints budget-error deltas:

```bash
python evals/budget_delta_experiment.py \
	--target-price 120 \
	--num-prompts 8 \
	--steps 20 \
	--save-images \
	--image-output-dir evals/budget_delta_images \
	--diffusion-source local-finetune \
	--oracle-checkpoint models/oracle/checkpoints/best.pt
```

HF variant:

```bash
python evals/budget_delta_experiment.py \
	--target-price 120 \
	--num-prompts 8 \
	--steps 20 \
	--save-images \
	--image-output-dir evals/budget_delta_images_hf \
	--diffusion-source hf \
	--hf-diffusion-model-id <your-hf-diffusion-model> \
	--oracle-checkpoint models/oracle/checkpoints/best.pt
```

When `--save-images` is enabled, each prompt saves:
- baseline image
- steered image
- side-by-side comparison image with prices/errors
- an HTML gallery at `index.html` in the image output directory

## Evaluation Metrics

### Steering Success Metrics

Compares three methods for price-controlled generation:

1. **My approach:** Steered diffusion (SD1.5/SDXL + LoRA + oracle steering)
2. **Baseline 1:** LoRA fine-tuned model without steering
3. **Baseline 2:** Vanilla base model (no LoRA, no steering)

### CLI Command

```bash
python evals/compute_steering_metrics.py \
  --csv data/knives_catalog.csv \
  --oracle-checkpoint models/oracle/checkpoints/best.pt \
  --lora-dir models/diffusion_lora \
  --num-samples 20 \
  --output evals/steering_results.json
```

### Options

- `--csv` — dataset CSV with prompts and prices
- `--prompt-columns` — which CSV columns to use for prompts (auto-detects: name, elements, why_love)
- `--price-column` — price column name (default: "price")
- `--oracle-checkpoint` — path to trained oracle
- `--lora-dir` — path to LoRA weights
- `--num-samples` — max samples to test (default: all)
- `--num-inference-steps` — diffusion steps (default: 30)
- `--output` — results JSON path (default: "evals/steering_results.json")

### Output

JSON with metrics for each method (mean error, median error, % within ±$15, ±$25, etc.):

```json
{
  "num_samples": 20,
  "steered": {
    "mean_error": 9.82,
    "median_error": 7.10,
    "pct_within_15": 78.0,
    "pct_within_25": 95.0
  },
  "lora_nosteering": {
    "mean_error": 28.40,
    "median_error": 24.20,
    "pct_within_15": 35.0,
    "pct_within_25": 60.0
  },
  "vanilla": {
    "mean_error": 42.30,
    "median_error": 38.50,
    "pct_within_15": 22.0,
    "pct_within_25": 40.0
  }
}
```

### FID to Real Images

Computes Fréchet Inception Distance between generated images and real product images using InceptionV3 features.

**Note:** InceptionV3 is a pre-trained feature extractor (loaded once at startup). Memory cost: ~1-2GB GPU RAM during extraction. Negligible overhead compared to diffusion generation.

#### CLI Command

```bash
python evals/compute_fid_metrics.py \
  --csv data/knives_catalog.csv \
  --image-root data/images \
  --oracle-checkpoint models/oracle/checkpoints/best.pt \
  --lora-dir models/diffusion_lora \
  --num-samples 20 \
  --output evals/fid_results.json
```

#### Options

- `--csv` — dataset CSV with prompts and prices
- `--image-root` — root directory containing real product images
- `--image-path-column` — CSV column with image paths (default: "image_path")
- `--prompt-columns` — which CSV columns to use for prompts (auto-detects)
- `--price-column` — price column name (default: "price")
- `--oracle-checkpoint` — path to trained oracle
- `--lora-dir` — path to LoRA weights
- `--num-samples` — max samples to test (default: all)
- `--num-inference-steps` — diffusion steps (default: 30)
- `--output` — results JSON path (default: "evals/fid_results.json")

#### Output

JSON with FID metrics for each method (lower FID = more similar to real images):

```json
{
  "num_samples": 20,
  "steered": {
    "mean_fid": 7.82,
    "median_fid": 6.45,
    "min_fid": 0.54,
    "max_fid": 28.12,
    "std_fid": 7.15
  },
  "lora_nosteering": {
    "mean_fid": 9.24,
    "median_fid": 8.10,
    "min_fid": 1.20,
    "max_fid": 35.40,
    "std_fid": 9.30
  },
  "vanilla": {
    "mean_fid": 12.30,
    "median_fid": 10.80,
    "min_fid": 2.10,
    "max_fid": 42.50,
    "std_fid": 11.20
  }
}
```

## Notes

- Bracelet-only filtering is supported in both oracle and diffusion training.
- CLIP is optional and baseline-only; core cost-guided flow is diffusion + oracle.
- If you want all categories instead of bracelets, use `--all-categories` on training scripts.