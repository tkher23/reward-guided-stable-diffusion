from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image

from inference.oracle_inference import OracleInference


def evaluate_price_alignment(
    oracle: OracleInference,
    images: list[Image.Image],
    target_price: float,
    tolerance_ratio: float = 0.1,
) -> Dict[str, float]:
    if len(images) == 0:
        return {
            "price_mean": 0.0,
            "price_std": 0.0,
            "price_mae_target": 0.0,
            "within_tolerance_rate": 0.0,
        }

    preds = oracle.score_pil_images(images).detach().cpu().numpy()
    abs_err = np.abs(preds - target_price)
    tol = target_price * tolerance_ratio
    within = (abs_err <= tol).astype(np.float32)

    return {
        "price_mean": float(np.mean(preds)),
        "price_std": float(np.std(preds)),
        "price_mae_target": float(np.mean(abs_err)),
        "within_tolerance_rate": float(np.mean(within)),
    }
