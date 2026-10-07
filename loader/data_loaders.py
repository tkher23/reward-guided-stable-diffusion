from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class ColumnConfig:
    image_path: str = "image_path"
    name: str = "name"
    elements: str = "elements"
    why_love: str = "why_love"
    dimensions: str = "dimensions"
    price: str = "price"
    product_id: Optional[str] = None


def default_prompt_builder(row: pd.Series, cols: ColumnConfig) -> str:
    name = str(row.get(cols.name, "") or "").strip()
    elements = str(row.get(cols.elements, "") or "").strip()
    why_love = str(row.get(cols.why_love, "") or "").strip()
    dimensions = str(row.get(cols.dimensions, "") or "").strip()
    price = str(row.get(cols.price, "") or "").strip()

    parts = []
    if name:
        parts.append(f"Name: {name}.")
    if elements:
        parts.append(f"Elements: {elements}.")
    if why_love:
        parts.append(f"Why you'll love it: {why_love}.")
    if dimensions:
        parts.append(f"Dimensions: {dimensions}.")
    if price:
        parts.append(f"Price: {price}.")
    return " ".join(parts)


class CatalogCLIPDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: str,
        columns: ColumnConfig,
        prompt_builder: Optional[Callable[[pd.Series, ColumnConfig], str]] = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_root = image_root
        self.columns = columns
        self.prompt_builder = prompt_builder or default_prompt_builder

        required_cols = [self.columns.image_path]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataframe: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        img_rel = str(row[self.columns.image_path])
        img_path = os.path.join(self.image_root, img_rel)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        text = self.prompt_builder(row, self.columns)

        if self.columns.product_id and self.columns.product_id in row:
            product_id = str(row[self.columns.product_id])
        else:
            product_id = str(idx)

        return {
            "image": image,
            "text": text,
            "product_id": product_id,
            "image_path": img_rel,
        }