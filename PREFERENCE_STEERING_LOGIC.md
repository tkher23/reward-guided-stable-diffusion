# Preference Steering System: Complete Logic & Architecture

## Overview

A system that learns **universal human preferences** from simulated user interactions, then uses those learned preferences to **personalize product generation in real-time**.

**Key insight:** Users have different tastes, but their preferences follow learnable patterns. We learn these patterns once (shared model θ), then efficiently personalize to each user (per-user vector u_A).

---

## The Core Problem

**Challenge:** How do you generate personalized product images without building a separate model for every user?

**Solution:**
1. Learn a **shared preference model (θ)** from aggregate user data
2. Represent each user efficiently as a **small weight vector (u_A)**
3. Combine them at generation time: **(θ + u_A) · features(image)**
4. Use gradient steering to steer diffusion toward high-preference images

---

## System Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    PREFERENCE MODEL                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  θ ∈ ℝ³              u_A ∈ ℝ³                              │
│  Shared preference   Per-user preference                   │
│  (learned once)      (updated per click)                    │
│                                                              │
│  θ = [w_colorful, w_chunky, w_clean]                        │
│  u_A = [w_colorful, w_chunky, w_clean] for user A          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE EXTRACTOR (φ)                      │
├─────────────────────────────────────────────────────────────┤
│  Differentiable CLIP-based feature extraction              │
│  For any shoe image x:                                      │
│    f(x) = [colorful_score, chunky_score, clean_score]      │
│    where each score ∈ [0, 1] via CLIP cosine similarity    │
│                                                              │
│  Examples:                                                  │
│    Bright colorful shoe → f(x) = [0.95, 0.10, 0.20]       │
│    Grey chunky shoe → f(x) = [0.15, 0.88, 0.25]           │
│    White minimal shoe → f(x) = [0.20, 0.05, 0.92]         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│              PREFERENCE SCORING FUNCTION                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  preference_score = (θ + u_A) · f(x)                       │
│                   = dot product in 3D feature space        │
│                                                              │
│  This is differentiable! ∇ score flows back through:       │
│    score → f(x) → CLIP → image → VAE → latent z_t        │
│                                                              │
│  Higher score = image matches this user's preferences      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│           DIFFUSION STEERING (per denoising step)          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Loss = -score  (maximize preference score)                │
│  ∇_{z_t} Loss steers latent toward high-preference images  │
│                                                              │
│  z_{t-1} ← z_{t-1} - η·∇ Loss                             │
│                                                              │
│  Result: Generated image aligns with (θ + u_A)            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│              USER PREFERENCE UPDATES (RL)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  After showing image to user:                              │
│                                                              │
│  IF user clicked:                                          │
│      u_A ← u_A + α · f(x)     (reinforce features)        │
│      θ ← θ + β · f(x)         (if adaptive mode)           │
│                                                              │
│  IF user skipped:                                          │
│      u_A ← u_A - α · f(x)     (punish features)           │
│      θ ← θ - β · f(x)         (if adaptive mode)           │
│                                                              │
│  Learning rates: α > β (user updates faster than shared)   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## The Three Feature Dimensions

The system operates in a **3-dimensional preference space**:

1. **Colorful** (0-1 score)
   - Colorful/bright sneaker → 0.9
   - Neutral/monochrome sneaker → 0.1
   - CLIP concept: `"colorful bright sneaker"`

2. **Chunky** (0-1 score)
   - Thick-soled platform shoe → 0.9
   - Slim low-profile shoe → 0.1
   - CLIP concept: `"chunky platform sneaker"`

3. **Clean** (0-1 score)
   - Minimal design, few logos → 0.9
   - Bold graphics, lots of details → 0.1
   - CLIP concept: `"clean minimal sneaker"`

---

## Phase 1: TRAINING (Learn Shared θ)

### Goal
Learn what human preferences look like in aggregate. Find patterns that work across many users.

### Process

**Setup:**
- 50 synthetic users created, each with a random true preference vector
- Example: User 1 = [0.8, 0.1, 0.2] (loves colorful, hates chunky, slightly prefers clean)

**Simulation (per user, 100 rounds):**
1. Randomly pick one of the 100 shoe images
2. Compute its feature vector: f(x) = [0.75, 0.15, 0.80]
3. Score how much this user would like it: score = θ·f(x)
4. User "clicks" if: score > threshold (with noise for realism)
5. Update θ based on click:
   - If clicked: θ ← θ + 0.01·f(x) (reinforce these features)
   - If skipped: θ ← θ - 0.01·f(x) (punish these features)

**After 50 users × 100 interactions:**
- θ has converged to an average human preference pattern
- Saved: `training_metadata.json` with final θ and all user preferences

### Output
- A shared θ that captures universal preference patterns
- Baseline: if you showed all shoes to a random person, θ would predict their average preferences

---

## Phase 2A: TESTING - FIXED THETA MODE

### Goal
Test if θ generalizes to new users that were **never seen during training**.

### Setup
- 3 held-out synthetic users (users 51, 52, 53)
- Each has their own TRUE preference (unknown to algorithm)
- Example: User 51 = [0.2, 0.7, 0.1] (loves chunky, hates colorful)

### Process (per held-out user, 50 rounds)

**Bootstrap phase (10 images):**
1. Show random shoes from the 100 images
2. Record which ones user 51 clicks
3. Learn their per-user vector: u_51 = average features of clicked shoes
   - If they clicked [0.2, 0.85, 0.15] and [0.1, 0.80, 0.20]
   - Then u_51 ≈ [0.15, 0.825, 0.175]

**Generation phase (40 rounds):**
For each round:
1. Generate personalized shoe using (θ + u_51) as steering target
2. Extract its features: f(x_generated)
3. Simulate: does user click?
   - They click if f(x_generated) aligns with their TRUE preference
4. Update u_51:
   - If clicked: u_51 ← u_51 + 0.05·f(x_generated)
   - If skipped: u_51 ← u_51 - 0.05·f(x_generated)
5. **CRITICAL: θ stays frozen** (not updated)

### Metrics
- **Click rate:** % of generated shoes user clicks (should increase over rounds)
- **Alignment:** cos_sim(learned u_51, true u_51) (should increase)
- **Convergence:** How quickly does u_51 learn the user's true preference?

### Output
- `test_results_fixed_theta.json`
- Per-user and aggregate click rates, alignment scores
- If click rates are high: θ generalizes! ✓

---

## Phase 2B: TESTING - ADAPTIVE THETA MODE

### Goal
Test if the model can **adapt in real-time** when new user preferences emerge (simulating trend changes).

### Setup
- Same 3 held-out users
- Same bootstrap + generation phases

### Key Difference
**θ is NOT frozen.** Both u_A and θ update:

```
Per click:
  u_51 ← u_51 + 0.05·f(x)    (user-specific learning)
  θ ← θ + 0.01·f(x)          (shared model adapts, slower rate)
```

### Why Slower θ Learning Rate?
- If one user drastically changes θ, it breaks the shared model for everyone
- β = 0.01 means 5 users over 50 rounds ≈ small update to θ
- This lets θ slowly adapt to emerging trends without instability

### Metrics
- Same as fixed mode
- **Additional:** How much did θ drift? (should be small, gradual changes)

### Comparison
- **Fixed θ:** Good baseline for generalization
- **Adaptive θ:** Better for real-world scenarios where preferences evolve
- **Delta (Adaptive - Fixed):** Shows value of continuous learning

### Output
- `test_results_adaptive_theta.json`
- Usually better click rates (θ adapts to trends)
- But θ should still stay somewhat stable

---

## Phase 3: EVALUATION

### Metrics Computed

**Per-user metrics:**
- `click_rate`: % of generated images user clicked (target: >60% with steering, ~30-40% random)
- `alignment`: cos_sim(learned_u_A, true_u_A) (target: >0.7)
- `convergence_speed`: rounds until alignment > 0.5

**Aggregate metrics:**
- `mean_click_rate` ± std
- `mean_alignment` ± std
- `improvement_delta` = adaptive - fixed (should be positive)

### Visualizations

**Figure 1: Convergence Comparison**
- X-axis: round number (0-50)
- Y-axis: click rate (%)
- Lines: per-user trajectories (moving average)
- Shows: does steering help over time?

**Figure 2: Preference Alignment**
- X-axis: user ID
- Y-axis: cos_sim(learned, true) [-1, 1]
- Points: fixed vs adaptive
- Shows: how well did we recover each user's true preference?

**Figure 3: Aggregate Report**
```
FIXED THETA MODE:
  Mean Click Rate: 55% ± 12%
  Mean Alignment: 0.68 ± 0.15

ADAPTIVE THETA MODE:
  Mean Click Rate: 62% ± 10%
  Mean Alignment: 0.72 ± 0.12

Improvement: +7 percentage points
```

---

## File Structure & Responsibilities

### Data Preparation
- **`data_prep/download_zappos.py`**: Stream 100 shoe images from Deep Lake
- **`data_prep/label_features_with_ai.py`**: Extract CLIP features for each image → JSON
- **`evals/visualize_data.py`**: Display images + features (eye-check)

### Core Models
- **`models/preference_oracle.py`**: PreferenceOracle class (θ storage + update logic)
- **`models/feature_extractor.py`**: DifferentiableFeatureExtractor (CLIP-based, differentiable)

### Inference
- **`inference/preference_oracle_inference.py`**: Wrapper combining oracle + extractor
- **`inference/preference_steering.py`**: Main steering loop (parallels diffusion_steering.py)

### Simulation
- **`simulation/synthetic_user_simulator.py`**: SyntheticUser class (has true_preference, clicks realistically)
- **`simulation/simulate_training_phase.py`**: Run 50 users × 100 interactions → learn θ
- **`simulation/simulate_test_phase.py`**: Run 3 held-out users × 50 rounds → test θ

### Evaluation
- **`evals/evaluate_preference_steering.py`**: Compute metrics, plot convergence, generate report

### Orchestration
- **`experiments/config.py`**: All hyperparameters (learning rates, feature concepts, etc.)
- **`experiments/full_preference_experiment.py`**: Main entry point, runs all phases sequentially

---

## How Features Are Extracted (CLIP-Based)

For each shoe image:

1. **Define 3 text concepts:**
   ```
   c1 = "colorful bright sneaker"
   c2 = "chunky platform sneaker"
   c3 = "clean minimal sneaker"
   ```

2. **Encode concepts via CLIP text encoder:**
   ```
   e1 = CLIP_text(c1)  # 512-dim vector
   e2 = CLIP_text(c2)
   e3 = CLIP_text(c3)
   ```

3. **Encode shoe image via CLIP image encoder:**
   ```
   img_embedding = CLIP_image(shoe_photo)  # 512-dim vector
   ```

4. **Compute cosine similarities:**
   ```
   f_colorful = cos_sim(img_embedding, e1)  ∈ [-1, 1]
   f_chunky = cos_sim(img_embedding, e2)
   f_clean = cos_sim(img_embedding, e3)
   ```

5. **Normalize to [0, 1]:**
   ```
   f(x) = [(f_colorful + 1)/2, (f_chunky + 1)/2, (f_clean + 1)/2]
   ```

**Why CLIP?**
- Pretrained on billions of web images + captions
- Understands semantic concepts ("colorful", "chunky", "clean")
- Differentiable (can backprop through it for steering)
- No training needed (zero-shot)

---

## How Steering Works (Differentiable Gradient Path)

At each denoising step:

```
z_t (latent at timestep t)
  ↓ [DDIM x̂_0 prediction]
x̂_0 (predicted clean image)
  ↓ [VAE decode]
image_tensor (pixel space, [B, 3, H, W])
  ↓ [CLIP image encoder]
img_embedding (512-dim vector)
  ↓ [cosine similarity with text embeddings]
f(x) = [f_colorful, f_chunky, f_clean] (3-dim)
  ↓ [dot product with preferences]
score = (θ + u_A) · f(x) (scalar)
  ↓ [loss]
L = -score (maximize score)
  ↓ [autograd backward]
∇_{z_t} L (gradient in latent space)
  ↓ [gradient descent step]
z_t ← z_t - η·∇_{z_t} L (steering update)
```

**All operations are differentiable** (except discrete latent sampling, which uses straight-through estimators if needed).

---

## Real-World Application (Future)

This system could enable:

1. **E-commerce customization:** User browses, sees personalized shoes generated in real-time
2. **Fashion retailers:** Show customer designs tailored to their taste + budget
3. **Trend forecasting:** Track how θ evolves over time → emerging aesthetic preferences
4. **A/B testing:** Compare fixed vs adaptive modes in production

---

## Key Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `theta_learning_rate` | 0.01 | How fast shared model learns (smaller = more stable) |
| `user_learning_rate` | 0.05 | How fast per-user preference learns (faster than θ) |
| `num_synthetic_train_users` | 50 | Diversity of training preferences |
| `interactions_per_train_user` | 100 | Training data volume per user |
| `num_synthetic_test_users` | 3 | Held-out users for evaluation |
| `rounds_per_test_user` | 50 | Rounds to measure convergence |
| `feature_concepts` | 3 | Colorful, Chunky, Clean |
| `steering_strength` | 1.0 | How aggressively gradients update latents |
| `guidance_scale` | 7.5 | CFG strength (higher = more adherence to prompt) |

---

## Success Criteria

✓ **Training phase succeeds if:**
- θ converges (stabilizes after ~2000 interactions)
- Click rates increase over users (learning is happening)

✓ **Test phase (fixed) succeeds if:**
- Click rates > 50% (better than random)
- Alignment > 0.6 (learned preferences match ground truth)
- Curves plateau (convergence achieved)

✓ **Test phase (adaptive) succeeds if:**
- Click rates > fixed θ click rates
- θ drifts slowly, doesn't collapse

✓ **Adaptive > Fixed if:**
- Adaptive click rate - Fixed click rate > 3-5 percentage points
- Shows value of continuous learning

---

## Edge Cases & Considerations

1. **Cold start:** First few rounds have poor u_A, so generated images might not match preferences. This is OK—algorithm is learning.

2. **Noise in clicks:** Simulated users click with probability (not deterministic). This is realistic but adds variance.

3. **Feature space quality:** If CLIP concepts don't align with visual features (e.g., "chunky" is ambiguous), the whole system suffers. Validation step checks this.

4. **Hyperparameter sensitivity:** Learning rates matter. Too high = instability. Too low = slow convergence.

5. **Scalability:** With more features (5D, 10D preference space), learning becomes harder but system becomes more expressive.

---

## Summary

**The preference steering system learns human preferences from simulated user data, then uses those preferences to personalize image generation via differentiable gradient steering.**

- **θ** = shared, learned-once model of human preferences
- **u_A** = efficient per-user personalization vector
- **φ** = differentiable feature extractor (CLIP-based)
- **Steering** = gradient of preference score backpropagates into latents
- **RL Update** = clicks update both θ (slowly) and u_A (quickly)
- **Evaluation** = compare fixed θ (baseline) vs adaptive θ (real-world scenario)
