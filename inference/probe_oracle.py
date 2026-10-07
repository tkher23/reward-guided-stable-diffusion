from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.oracle_inference import OracleInference


def _resolve_existing_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p

    candidates = [Path.cwd() / p, PROJECT_ROOT / p]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(f"Path not found: {path_str}")


def _resolve_image_from_row(row: pd.Series, image_root: Path) -> Path:
    rel = str(row.get("image_path", "")).strip()
    if not rel:
        raise ValueError("Matched CSV row has no image_path value.")

    img_path = Path(rel)
    if img_path.is_absolute() and img_path.exists():
        return img_path

    resolved = image_root / img_path
    if resolved.exists():
        return resolved

    raise FileNotFoundError(f"Image path from CSV not found: {resolved}")


def _find_row_by_id(df: pd.DataFrame, image_id: str) -> pd.Series:
    if "id" not in df.columns:
        raise ValueError("CSV does not contain an 'id' column.")

    id_as_str = image_id.strip()
    id_series = df["id"].astype(str).str.strip()
    matches = df[id_series == id_as_str]
    if matches.empty:
        raise ValueError(f"No row found for id='{image_id}'.")
    return matches.iloc[0]


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        s = str(value).strip().replace(",", "")
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def run_probe(
    oracle_checkpoint: str,
    image_root: str,
    csv_path: str,
    image_id: str | None,
    image_path: str | None,
    description: str,
    device: str,
) -> dict[str, Any]:
    oracle_ckpt = _resolve_existing_path(oracle_checkpoint)
    img_root = _resolve_existing_path(image_root)

    row = None
    resolved_image_path: Path | None = None

    if image_path:
        resolved_image_path = _resolve_existing_path(image_path)

    if image_id:
        resolved_csv = _resolve_existing_path(csv_path)
        df = pd.read_csv(resolved_csv)
        row = _find_row_by_id(df, image_id)
        if resolved_image_path is None:
            resolved_image_path = _resolve_image_from_row(row, img_root)

    if resolved_image_path is None:
        raise ValueError("Provide either --image-path or --image-id (with resolvable CSV/image_root).")

    img = Image.open(resolved_image_path).convert("RGB")

    oracle = OracleInference(str(oracle_ckpt), device=device)
    pred_price_tensor = oracle.score_pil_images([img])
    pred_price = float(pred_price_tensor[0].item())
    pred_log = float(torch.log(pred_price_tensor[0].clamp_min(1e-8)).item())

    result: dict[str, Any] = {
        "image_path": str(resolved_image_path),
        "description_input": description,
        "predicted_price": pred_price,
        "predicted_log_price": pred_log,
        "note": "Oracle uses image only; description is included for your reference.",
    }

    if row is not None:
        actual_price = _to_float_or_none(row.get("price"))
        result["id"] = str(row.get("id", image_id))
        result["csv_image_path"] = str(row.get("image_path", ""))
        result["csv_name"] = str(row.get("name", row.get("product_name", "")))
        result["csv_elements"] = str(row.get("elements", ""))
        result["csv_why_love"] = str(row.get("why_love", row.get("why_youll_love_it", "")))
        result["csv_price"] = actual_price
        if actual_price is not None:
            signed_error = pred_price - actual_price
            result["signed_error_vs_csv_price"] = signed_error
            result["abs_error_vs_csv_price"] = abs(signed_error)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe oracle prediction for a single image (by id or path)."
    )
    parser.add_argument("--oracle-checkpoint", type=str, default="models/oracle/checkpoints/best.pt")
    parser.add_argument("--image-root", type=str, default="data/images")
    parser.add_argument("--csv", type=str, default="data/catalog_clip_ready.csv")
    parser.add_argument("--image-id", type=str, default="", help="Row id from CSV")
    parser.add_argument("--image-path", type=str, default="", help="Direct path to image")
    parser.add_argument("--description", type=str, default="", help="Description text for your manual check")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    result = run_probe(
        oracle_checkpoint=args.oracle_checkpoint,
        image_root=args.image_root,
        csv_path=args.csv,
        image_id=args.image_id or None,
        image_path=args.image_path or None,
        description=args.description,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
