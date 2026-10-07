from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.diffusion_steering import SteeringConfig
from inference.generate_designs import DesignInference


def resolve_existing_path(path_str: str) -> str:
    p = Path(path_str)
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, PROJECT_ROOT / p]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return ""


def resolve_csv_path(csv_path: str) -> str:
    resolved = resolve_existing_path(csv_path)
    if resolved:
        return resolved

    fallback_candidates = [
        PROJECT_ROOT / "data" / "all_products.csv",
        PROJECT_ROOT / "data" / "all_products_partial.csv",
    ]
    for c in fallback_candidates:
        if c.exists():
            print(
                f"CSV '{csv_path}' not found. Falling back to '{c.as_posix()}'."
            )
            return str(c.resolve())

    data_dir = PROJECT_ROOT / "data"
    available = []
    if data_dir.exists():
        available = sorted([p.name for p in data_dir.glob("*.csv")])
    raise FileNotFoundError(
        "Could not resolve CSV path. Tried current working directory and repo root. "
        f"Requested: '{csv_path}'. Available CSVs in data/: {available}"
    )


def build_prompt(row: pd.Series) -> str:
    name = str(row.get("name", "") or "").strip()
    elements = str(row.get("elements", "") or "").strip()
    why_love = str(row.get("why_love", "") or "").strip()

    parts = []
    if name:
        parts.append(name)
    if elements:
        parts.append(f"Elements: {elements}")
    if why_love:
        parts.append(why_love)
    return ". ".join(parts).strip() or "bracelet jewelry design"


def load_bracelet_prompts(csv_path: str, num_prompts: int) -> list[str]:
    resolved_csv = resolve_csv_path(csv_path)
    print(f"Resolved csv path: {resolved_csv}")
    df = pd.read_csv(resolved_csv)
    if "image_path" in df.columns:
        df = df[df["image_path"].astype(str).str.startswith("bracelets/")]
    df = df.reset_index(drop=True)

    prompts = [build_prompt(row) for _, row in df.head(num_prompts).iterrows()]
    return prompts


def generate_baseline_image(engine: DesignInference, prompt: str, steps: int, guidance_scale: float):
    return engine.diffusion.generate_batch(
        [prompt],
        num_images_per_prompt=1,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    )[0]


def score_price(engine: DesignInference, image) -> float:
    if engine.oracle is None:
        raise RuntimeError("Oracle is required for budget-delta experiments.")
    return float(engine.oracle.score_pil_images([image]).item())


def _slugify_prompt(prompt: str, max_len: int = 60) -> str:
    keep = []
    for ch in prompt.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_"}:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return (slug or "prompt")[:max_len]


def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def save_comparison_images(
    baseline_img: Image.Image,
    steered_img: Image.Image,
    out_dir: Path,
    index: int,
    prompt: str,
    baseline_price: float,
    steered_price: float,
    baseline_err: float,
    steered_err: float,
    delta: float,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{index:03d}_{_slugify_prompt(prompt)}"

    baseline_path = out_dir / f"{stem}_baseline.png"
    steered_path = out_dir / f"{stem}_steered.png"
    compare_path = out_dir / f"{stem}_comparison.png"

    _ensure_rgb(baseline_img).save(baseline_path)
    _ensure_rgb(steered_img).save(steered_path)

    left = _ensure_rgb(baseline_img)
    right = _ensure_rgb(steered_img)
    panel_w = max(left.width, right.width)
    panel_h = max(left.height, right.height)
    header_h = 88

    canvas = Image.new("RGB", (panel_w * 2, panel_h + header_h), color=(250, 250, 250))
    canvas.paste(left.resize((panel_w, panel_h)), (0, header_h))
    canvas.paste(right.resize((panel_w, panel_h)), (panel_w, header_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"Prompt: {prompt[:120]}", fill=(20, 20, 20))
    draw.text(
        (12, 30),
        f"Baseline price={baseline_price:.2f} err={baseline_err:.2f} | "
        f"Steered price={steered_price:.2f} err={steered_err:.2f} | Delta={delta:+.2f}",
        fill=(20, 20, 20),
    )
    draw.text((12, 54), "Left: baseline (no steering)", fill=(60, 60, 60))
    draw.text((panel_w + 12, 54), "Right: steered", fill=(60, 60, 60))

    canvas.save(compare_path)
    return {
        "baseline_image": str(baseline_path).replace("\\", "/"),
        "steered_image": str(steered_path).replace("\\", "/"),
        "comparison_image": str(compare_path).replace("\\", "/"),
    }


def write_gallery_html(rows: list[dict], html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)

    cards = []
    for r in rows:
        comp = r.get("comparison_image", "")
        if not comp:
            continue
        cards.append(
            (
                "<div class='card'>"
                f"<h3>#{r['index']} | Delta {r['budget_error_delta']:+.2f}</h3>"
                f"<p>{r['prompt']}</p>"
                f"<img src='{Path(comp).name}' alt='comparison {r['index']}' />"
                "</div>"
            )
        )

    html = """
<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Budget Delta Gallery</title>
  <style>
    body { font-family: -apple-system, Segoe UI, sans-serif; margin: 20px; background: #f6f7fb; }
    .grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); }
    .card { background: white; border-radius: 12px; padding: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
    img { width: 100%; height: auto; border-radius: 8px; }
    h3 { margin: 6px 0; }
    p { color: #444; margin-top: 4px; }
  </style>
</head>
<body>
  <h1>Budget Error Delta Visual Gallery</h1>
  <div class='grid'>
    __CARDS__
  </div>
</body>
</html>
""".strip()

    html = html.replace("__CARDS__", "\n".join(cards))
    html_path.write_text(html, encoding="utf-8")


def run_budget_delta_experiment(
    engine: DesignInference,
    prompts: list[str],
    target_price: float,
    steps: int,
    guidance_scale: float,
    steering_strength: float,
    steer_every_k_steps: int,
    seed: int,
    save_images: bool,
    image_output_dir: str,
) -> dict:
    rows = []
    img_dir = Path(image_output_dir)

    for i, prompt in enumerate(prompts, 1):
        run_seed = seed + i

        torch.manual_seed(run_seed)
        baseline_img = generate_baseline_image(engine, prompt, steps=steps, guidance_scale=guidance_scale)
        baseline_price = score_price(engine, baseline_img)
        baseline_err = abs(baseline_price - target_price)

        torch.manual_seed(run_seed)
        steer_cfg = SteeringConfig(
            target_price=target_price,
            steering_strength=steering_strength,
            steer_every_k_steps=steer_every_k_steps,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            debug_log=False,
        )
        steered_img = engine.steered_generator.generate(prompt, steer_cfg)
        steered_price = score_price(engine, steered_img)
        steered_err = abs(steered_price - target_price)

        delta = baseline_err - steered_err
        row = {
            "index": i,
            "prompt": prompt,
            "baseline_price": baseline_price,
            "steered_price": steered_price,
            "baseline_error": baseline_err,
            "steered_error": steered_err,
            "budget_error_delta": delta,
        }

        if save_images:
            row.update(
                save_comparison_images(
                    baseline_img=baseline_img,
                    steered_img=steered_img,
                    out_dir=img_dir,
                    index=i,
                    prompt=prompt,
                    baseline_price=baseline_price,
                    steered_price=steered_price,
                    baseline_err=baseline_err,
                    steered_err=steered_err,
                    delta=delta,
                )
            )

        rows.append(row)

        print(
            f"[{i}/{len(prompts)}] "
            f"base_err={baseline_err:.2f} steered_err={steered_err:.2f} delta={delta:+.2f}"
        )

    df = pd.DataFrame(rows)
    summary = {
        "count": int(len(df)),
        "target_price": float(target_price),
        "mean_baseline_error": float(df["baseline_error"].mean()) if len(df) else 0.0,
        "mean_steered_error": float(df["steered_error"].mean()) if len(df) else 0.0,
        "mean_budget_error_delta": float(df["budget_error_delta"].mean()) if len(df) else 0.0,
        "median_budget_error_delta": float(df["budget_error_delta"].median()) if len(df) else 0.0,
        "steering_win_rate": float((df["budget_error_delta"] > 0).mean()) if len(df) else 0.0,
    }

    print("\n=== Budget Error Delta Summary ===")
    print(json.dumps(summary, indent=2))

    if save_images:
        gallery_path = img_dir / "index.html"
        write_gallery_html(rows, gallery_path)
        print(f"Saved image gallery to {gallery_path}")

    return {
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B experiment: budget-error delta (no-steer vs steer)")
    parser.add_argument("--csv", type=str, default="data/catalog_clip_ready.csv", help="Prompt source CSV")
    parser.add_argument("--num-prompts", type=int, default=8, help="Number of bracelet prompts to test")
    parser.add_argument("--target-price", type=float, required=True, help="Target budget in USD")
    parser.add_argument("--steps", type=int, default=20, help="Diffusion inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--steering-strength", type=float, default=0.08, help="Steering update strength")
    parser.add_argument("--steer-every-k-steps", type=int, default=1, help="Apply steering every k denoising steps")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed for paired runs")
    parser.add_argument("--output-json", type=str, default="evals/budget_delta_report.json", help="Where to save experiment report")
    parser.add_argument("--save-images", action="store_true", help="Save baseline/steered/comparison images during eval")
    parser.add_argument("--image-output-dir", type=str, default="evals/budget_delta_images", help="Folder for saved eval images")

    parser.add_argument("--lora-dir", type=str, default="models/diffusion_lora", help="Local LoRA directory")
    parser.add_argument("--oracle-checkpoint", type=str, default="models/oracle/checkpoints/best.pt", help="Oracle checkpoint path")
    parser.add_argument("--diffusion-source", choices=["hf", "local-finetune"], default="local-finetune")
    parser.add_argument("--hf-diffusion-model-id", type=str, default="", help="HF model id if diffusion-source=hf")

    args = parser.parse_args()

    prompts = load_bracelet_prompts(args.csv, args.num_prompts)
    if not prompts:
        raise RuntimeError("No bracelet prompts found. Check csv path/content.")

    engine = DesignInference(
        diffusion_lora_dir=args.lora_dir,
        oracle_checkpoint=args.oracle_checkpoint,
        use_clip_eval=False,
        enable_steering=True,
        diffusion_source=args.diffusion_source,
        hf_diffusion_model_id=args.hf_diffusion_model_id or None,
        train_local_diffusion_first=False,
    )

    if engine.oracle is None or engine.steered_generator is None:
        raise RuntimeError("Steering experiment requires loaded oracle and steered generator.")

    report = run_budget_delta_experiment(
        engine=engine,
        prompts=prompts,
        target_price=args.target_price,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        steering_strength=args.steering_strength,
        steer_every_k_steps=args.steer_every_k_steps,
        seed=args.seed,
        save_images=args.save_images,
        image_output_dir=args.image_output_dir,
    )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {out}")


if __name__ == "__main__":
    main()
