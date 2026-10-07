# python training/train_clip.py

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loader.data_loaders import CatalogCLIPDataset, ColumnConfig
from evals.evaluate_clip import evaluate_retrieval
from models.model_clip import CLIPFinetune, FreezeConfig, count_trainable_parameters, freeze_lower_layers


@dataclass
class TrainConfig:
    csv_path: str = "data/catalog_clip_ready.csv"
    image_root: str = "data/images"
    output_dir: str = "models/clip_baseline"

    model_name: str = "openai/clip-vit-base-patch32"
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 1e-2
    seed: int = 42

    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1

    freeze_vision_embeddings: bool = True
    freeze_text_embeddings: bool = True
    freeze_vision_layers: int = 8
    freeze_text_layers: int = 8

    columns: dict = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_output_dirs(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "checkpoints").mkdir(parents=True, exist_ok=True)
    Path(output_dir, "eval").mkdir(parents=True, exist_ok=True)


def split_dataframe(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-6):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

    return train_df, val_df, test_df


def build_collate_fn(processor: CLIPProcessor):
    def collate_fn(batch: list[dict]) -> dict:
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        product_ids = [item["product_id"] for item in batch]
        image_paths = [item["image_path"] for item in batch]

        encoded = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded["product_id"] = product_ids
        encoded["image_path"] = image_paths
        encoded["raw_text"] = texts
        return encoded

    return collate_fn


def train_one_epoch(model, loader, optimizer, device: str) -> float:
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=True,
        )
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate_loss(model, loader, device: str) -> float:
    model.eval()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Validating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=True,
        )
        running_loss += outputs.loss.item()

    return running_loss / max(len(loader), 1)


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, save_path)


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    cfg = TrainConfig(
        columns={
            "image_path": "image_path",
            "name": "name",
            "elements": "elements",
            "why_love": "why_love",
            "dimensions": "dimensions",
            "price": "price",
            "product_id": "id",
        }
    )

    # On Windows, DataLoader worker spawn requires picklable callables.
    # This script uses a local collate closure, so force single-process loading.
    if os.name == "nt" and cfg.num_workers > 0:
        print("Windows detected: forcing num_workers=0 to avoid DataLoader pickling errors.")
        cfg.num_workers = 0

    set_seed(cfg.seed)
    make_output_dirs(cfg.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(cfg.model_name)

    df = pd.read_csv(cfg.csv_path)
    train_df, val_df, test_df = split_dataframe(
        df,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
    )

    # Save splits for reproducibility
    train_df.to_csv(os.path.join(cfg.output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(cfg.output_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(cfg.output_dir, "test_split.csv"), index=False)

    columns = ColumnConfig(**cfg.columns)

    train_ds = CatalogCLIPDataset(train_df, cfg.image_root, columns)
    val_ds = CatalogCLIPDataset(val_df, cfg.image_root, columns)
    test_ds = CatalogCLIPDataset(test_df, cfg.image_root, columns)

    collate_fn = build_collate_fn(processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = CLIPFinetune(model_name=cfg.model_name)
    freeze_cfg = FreezeConfig(
        freeze_vision_embeddings=cfg.freeze_vision_embeddings,
        freeze_text_embeddings=cfg.freeze_text_embeddings,
        freeze_vision_layers=cfg.freeze_vision_layers,
        freeze_text_layers=cfg.freeze_text_layers,
    )
    freeze_lower_layers(model, freeze_cfg)
    model = model.to(device)

    param_counts = count_trainable_parameters(model)
    print("Parameter counts:", param_counts)

    save_json(
        {
            "train_config": asdict(cfg),
            "freeze_config": asdict(freeze_cfg),
            "parameter_counts": param_counts,
            "device": device,
        },
        os.path.join(cfg.output_dir, "run_config.json"),
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(cfg.output_dir, "checkpoints", "best.pt")
    last_ckpt_path = os.path.join(cfg.output_dir, "checkpoints", "last.pt")

    history = []

    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_loss(model, val_loader, device)
        val_metrics = evaluate_retrieval(model, val_loader, device)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(epoch_metrics)

        print(json.dumps(epoch_metrics, indent=2))

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics=epoch_metrics,
            save_path=last_ckpt_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=epoch_metrics,
                save_path=best_ckpt_path,
            )
            print(f"Saved new best checkpoint to {best_ckpt_path}")

        save_json(
            {"history": history},
            os.path.join(cfg.output_dir, "train_history.json"),
        )

    # Load best model for final test evaluation
    best_payload = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_payload["model_state_dict"])

    test_metrics = evaluate_retrieval(model, test_loader, device)
    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))
    save_json(test_metrics, os.path.join(cfg.output_dir, "eval", "test_metrics.json"))


if __name__ == "__main__":
    main()