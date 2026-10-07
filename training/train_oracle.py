from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet18_Weights
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_oracle import PriceOracleResNet18


@dataclass
class OracleTrainConfig:
    csv_path: str = "data/catalog_clip_ready.csv"
    image_root: str = "data/images"
    output_dir: str = "models/oracle"

    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    image_size: int = 224
    seed: int = 42
    bracelets_only: bool = True
    cv_folds: int = 1
    price_bins: int = 5


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


def resolve_output_dir(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def parse_price(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if not match:
        return None
    p = float(match.group(0))
    if p <= 0:
        return None
    return p


class OraclePriceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, image_size: int = 224, bracelets_only: bool = True):
        self.image_root = image_root
        self.df = df.reset_index(drop=True).copy()
        if bracelets_only and "image_path" in self.df.columns:
            self.df = self.df[self.df["image_path"].astype(str).str.startswith("bracelets/")].reset_index(drop=True)
        self.df["price_numeric"] = self.df["price"].apply(parse_price)
        self.df = self.df[self.df["price_numeric"].notna()].reset_index(drop=True)
        self.df["log_price"] = np.log(self.df["price_numeric"].astype(float))

        weights = ResNet18_Weights.DEFAULT
        self.normalize = weights.transforms()
        self.resize = v2.Resize((image_size, image_size), antialias=True)
        self.to_tensor = v2.ToImage()
        self.to_dtype = v2.ToDtype(torch.float32, scale=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_rel = str(row["image_path"])
        img_path = os.path.join(self.image_root, img_rel)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        x = self.to_dtype(self.to_tensor(img))
        x = self.resize(x)
        x = self.normalize(x)

        return {
            "image": x,
            "log_price": torch.tensor(float(row["log_price"]), dtype=torch.float32),
            "price": torch.tensor(float(row["price_numeric"]), dtype=torch.float32),
            "id": str(row.get("id", idx)),
        }


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


def make_price_bins(log_prices: pd.Series, n_bins: int) -> np.ndarray:
    arr = np.asarray(log_prices, dtype=np.float64)
    n_unique = int(np.unique(arr).shape[0])
    if n_unique <= 1:
        return np.zeros_like(arr, dtype=np.int64)

    bins = max(2, min(int(n_bins), n_unique))
    binned = pd.qcut(arr, q=bins, labels=False, duplicates="drop")
    if isinstance(binned, pd.Series):
        out = binned.fillna(0).astype(int).to_numpy()
    else:
        out = pd.Series(binned).fillna(0).astype(int).to_numpy()
    return out


def _stratified_take(
    indices: list[int],
    bins: np.ndarray,
    ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    keep = []
    take = []

    by_bin: dict[int, list[int]] = {}
    for idx in indices:
        b = int(bins[idx])
        by_bin.setdefault(b, []).append(idx)

    for b, idxs in by_bin.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_take = int(round(n * ratio))
        if ratio > 0 and n >= 3:
            n_take = max(1, n_take)
        n_take = min(max(n_take, 0), max(n - 1, 0))

        take.extend(idxs[:n_take])
        keep.extend(idxs[n_take:])

    return sorted(keep), sorted(take)


def stratified_split_indices(
    bins: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    all_idx = list(range(len(bins)))
    train_val_idx, test_idx = _stratified_take(all_idx, bins, test_ratio, seed)
    train_idx, val_idx = _stratified_take(train_val_idx, bins, val_ratio, seed + 17)
    return train_idx, val_idx, test_idx


def make_stratified_kfolds(
    bins: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]

    by_bin: dict[int, list[int]] = {}
    for i, b in enumerate(bins):
        by_bin.setdefault(int(b), []).append(i)

    for _, idxs in by_bin.items():
        rng.shuffle(idxs)
        for i, idx in enumerate(idxs):
            folds[i % n_folds].append(idx)

    return [sorted(f) for f in folds]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    medae = float(np.median(np.abs(err)))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "medae": medae,
    }


def run_epoch(model, loader, criterion, optimizer, device: str, train: bool) -> dict[str, float]:
    model.train(mode=train)
    ys = []
    yhs = []
    loss_accum = 0.0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["log_price"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        yhat = model(x)
        loss = criterion(yhat, y)

        if train:
            loss.backward()
            optimizer.step()

        loss_accum += float(loss.item()) * x.size(0)
        ys.append(y.detach().cpu().numpy())
        yhs.append(yhat.detach().cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.float32)
    y_pred = np.concatenate(yhs) if yhs else np.array([], dtype=np.float32)
    m = regression_metrics(y_true, y_pred) if len(y_true) else {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "medae": 0.0}
    m["loss"] = float(loss_accum / max(len(loader.dataset), 1))
    return m


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def train_single_split(
    dataset: OraclePriceDataset,
    cfg: OracleTrainConfig,
    out_dir: Path,
    tr_idx: list[int],
    va_idx: list[int],
    te_idx: list[int],
    fold_name: str,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_ds = Subset(dataset, tr_idx)
    va_ds = Subset(dataset, va_idx)
    te_ds = Subset(dataset, te_idx)

    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    dataset.df.iloc[tr_idx].to_csv(split_dir / "train.csv", index=False)
    dataset.df.iloc[va_idx].to_csv(split_dir / "val.csv", index=False)
    dataset.df.iloc[te_idx].to_csv(split_dir / "test.csv", index=False)

    effective_workers = max(int(cfg.num_workers), 0)
    pin_memory = bool(cfg.pin_memory and device == "cuda")
    persistent_workers = bool(cfg.persistent_workers and effective_workers > 0)

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": effective_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if effective_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.prefetch_factor)

    print(
        f"[{fold_name}] Dataloader config: "
        f"workers={effective_workers}, pin_memory={pin_memory}, "
        f"persistent_workers={persistent_workers}, "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')}"
    )

    tr_loader = DataLoader(tr_ds, shuffle=True, **loader_kwargs)
    va_loader = DataLoader(va_ds, shuffle=False, **loader_kwargs)
    te_loader = DataLoader(te_ds, shuffle=False, **loader_kwargs)

    model = PriceOracleResNet18(pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    best_r2 = -math.inf
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        train_m = run_epoch(model, tr_loader, criterion, optimizer, device, train=True)
        val_m = run_epoch(model, va_loader, criterion, optimizer, device, train=False)

        row = {
            "epoch": epoch,
            "train": train_m,
            "val": val_m,
        }
        history.append(row)
        print(f"[{fold_name}] " + json.dumps(row, indent=2))

        if val_m["r2"] > best_r2:
            best_r2 = val_m["r2"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "best_val_r2": best_r2,
                    "image_size": cfg.image_size,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = run_epoch(model, te_loader, criterion, optimizer, device, train=False)

    summary = {
        "fold": fold_name,
        "best_val_r2": best_r2,
        "test": test_m,
        "train_size": len(tr_ds),
        "val_size": len(va_ds),
        "test_size": len(te_ds),
    }

    save_json(asdict(cfg), str(out_dir / "train_config.json"))
    save_json({"history": history}, str(out_dir / "train_history.json"))
    save_json(summary, str(out_dir / "test_metrics.json"))
    print(f"[{fold_name}] " + json.dumps(summary, indent=2))
    return summary


def train_oracle(cfg: OracleTrainConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    csv_path = resolve_csv_path(cfg.csv_path)
    image_root = resolve_existing_path(cfg.image_root)
    if not image_root:
        raise FileNotFoundError(
            f"Could not resolve image root '{cfg.image_root}' from cwd or repo root."
        )
    output_dir = resolve_output_dir(cfg.output_dir)

    print(f"Resolved csv path: {csv_path}")
    print(f"Resolved image root: {image_root}")
    print(f"Resolved output dir: {output_dir}")

    out_dir = Path(output_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    dataset = OraclePriceDataset(
        df,
        image_root,
        cfg.image_size,
        bracelets_only=cfg.bracelets_only,
    )
    print(f"Oracle dataset size (valid priced rows): {len(dataset)}")

    bins = make_price_bins(dataset.df["log_price"], cfg.price_bins)
    print(
        "Price-bin distribution: "
        + json.dumps({int(k): int(v) for k, v in zip(*np.unique(bins, return_counts=True))})
    )

    if cfg.cv_folds <= 1:
        tr_idx, va_idx, te_idx = stratified_split_indices(
            bins,
            cfg.val_ratio,
            cfg.test_ratio,
            cfg.seed,
        )
        summary = train_single_split(
            dataset=dataset,
            cfg=cfg,
            out_dir=out_dir,
            tr_idx=tr_idx,
            va_idx=va_idx,
            te_idx=te_idx,
            fold_name="single_split",
        )
        print(json.dumps(summary, indent=2))
        return

    n_samples = len(dataset)
    if cfg.cv_folds > n_samples:
        raise ValueError(
            f"cv_folds={cfg.cv_folds} cannot exceed dataset size={n_samples}."
        )

    folds = make_stratified_kfolds(bins, cfg.cv_folds, cfg.seed)
    all_idx = set(range(n_samples))
    fold_summaries = []

    for i, te_idx in enumerate(folds, start=1):
        fold_dir = out_dir / "folds" / f"fold_{i}"
        te_set = set(te_idx)
        train_val_idx = sorted(list(all_idx - te_set))
        tr_idx, va_idx = _stratified_take(
            train_val_idx,
            bins,
            cfg.val_ratio,
            cfg.seed + i * 101,
        )

        print(
            f"Fold {i}/{cfg.cv_folds}: "
            f"train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}"
        )
        fold_summary = train_single_split(
            dataset=dataset,
            cfg=cfg,
            out_dir=fold_dir,
            tr_idx=tr_idx,
            va_idx=va_idx,
            te_idx=te_idx,
            fold_name=f"fold_{i}",
        )
        fold_summaries.append(fold_summary)

    test_r2 = np.array([f["test"]["r2"] for f in fold_summaries], dtype=np.float32)
    test_mae = np.array([f["test"]["mae"] for f in fold_summaries], dtype=np.float32)
    test_rmse = np.array([f["test"]["rmse"] for f in fold_summaries], dtype=np.float32)
    best_val_r2 = np.array([f["best_val_r2"] for f in fold_summaries], dtype=np.float32)

    cv_summary = {
        "cv_folds": int(cfg.cv_folds),
        "price_bins": int(cfg.price_bins),
        "mean_best_val_r2": float(best_val_r2.mean()),
        "std_best_val_r2": float(best_val_r2.std()),
        "mean_test_r2": float(test_r2.mean()),
        "std_test_r2": float(test_r2.std()),
        "mean_test_mae": float(test_mae.mean()),
        "std_test_mae": float(test_mae.std()),
        "mean_test_rmse": float(test_rmse.mean()),
        "std_test_rmse": float(test_rmse.std()),
        "folds": fold_summaries,
    }

    save_json(asdict(cfg), str(out_dir / "train_config.json"))
    save_json(cv_summary, str(out_dir / "cv_metrics.json"))
    print("Cross-validation summary:")
    print(json.dumps(cv_summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train clean cost oracle (ResNet18 regression)")
    parser.add_argument("--csv", type=str, default="data/catalog_clip_ready.csv", help="Input CSV path")
    parser.add_argument("--image-root", type=str, default="data/images", help="Root image folder")
    parser.add_argument("--output", type=str, default="models/oracle", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--pin-memory", action="store_true", help="Enable pinned CPU memory for faster H2D transfer")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pinned CPU memory")
    parser.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive between epochs")
    parser.add_argument("--no-persistent-workers", action="store_true", help="Disable persistent DataLoader workers")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched per worker when num_workers > 0")
    parser.add_argument("--cv-folds", type=int, default=1, help="Number of stratified CV folds (set 5 for 5-fold CV)")
    parser.add_argument("--price-bins", type=int, default=5, help="Number of price quantile bins for stratified sampling")
    parser.add_argument("--bracelets-only", action="store_true", help="Train only on rows with image_path under bracelets/")
    parser.add_argument("--all-categories", action="store_true", help="Disable bracelet-only filter")
    args = parser.parse_args()

    bracelets_only = True
    if args.all_categories:
        bracelets_only = False
    elif args.bracelets_only:
        bracelets_only = True

    pin_memory = True
    if args.no_pin_memory:
        pin_memory = False
    elif args.pin_memory:
        pin_memory = True

    persistent_workers = True
    if args.no_persistent_workers:
        persistent_workers = False
    elif args.persistent_workers:
        persistent_workers = True

    cfg = OracleTrainConfig(
        csv_path=args.csv,
        image_root=args.image_root,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=max(args.num_workers, 0),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=max(args.prefetch_factor, 1),
        cv_folds=max(args.cv_folds, 1),
        price_bins=max(args.price_bins, 2),
        bracelets_only=bracelets_only,
    )
    train_oracle(cfg)
