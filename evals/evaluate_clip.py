from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def encode_dataset(model, loader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor, list]:
    model.eval()

    all_image_embeds = []
    all_text_embeds = []
    all_ids = []

    for batch in tqdm(loader, desc="Encoding eval set", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        image_embeds = model.get_image_embeddings(pixel_values=pixel_values)
        text_embeds = model.get_text_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        all_image_embeds.append(image_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())
        all_ids.extend(batch["product_id"])

    return (
        torch.cat(all_image_embeds, dim=0),
        torch.cat(all_text_embeds, dim=0),
        all_ids,
    )


def compute_recall_at_k(similarity: torch.Tensor, product_ids: list, ks: list[int]) -> Dict[str, float]:
    """
    Compute recall using product_id matching (not diagonal).
    similarity: [N, N] where N = number of items
    product_ids: list of N product IDs
    """
    n = similarity.size(0)
    results = {}

    # Build ID-to-index mapping
    id_to_indices = {}
    for idx, pid in enumerate(product_ids):
        if pid not in id_to_indices:
            id_to_indices[pid] = []
        id_to_indices[pid].append(idx)

    # image -> text: for each image row, find where its matching text(s) rank
    img_ranks = similarity.argsort(dim=1, descending=True)
    img_match_ranks = []
    for i in range(n):
        query_id = product_ids[i]
        target_indices = id_to_indices[query_id]
        # Find the best (lowest) rank among all matching targets
        ranks_for_query = []
        for target_idx in target_indices:
            rank = (img_ranks[i] == target_idx).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                ranks_for_query.append(rank[0].item())
        if ranks_for_query:
            img_match_ranks.append(min(ranks_for_query))

    # text -> image: for each text column, find where its matching image(s) rank
    txt_ranks = similarity.argsort(dim=0, descending=True).t()
    txt_match_ranks = []
    for i in range(n):
        query_id = product_ids[i]
        target_indices = id_to_indices[query_id]
        ranks_for_query = []
        for target_idx in target_indices:
            rank = (txt_ranks[i] == target_idx).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                ranks_for_query.append(rank[0].item())
        if ranks_for_query:
            txt_match_ranks.append(min(ranks_for_query))

    img_match_ranks = torch.tensor(img_match_ranks)
    txt_match_ranks = torch.tensor(txt_match_ranks)

    for k in ks:
        results[f"image_to_text_R@{k}"] = (img_match_ranks < k).float().mean().item()
        results[f"text_to_image_R@{k}"] = (txt_match_ranks < k).float().mean().item()

    results["mean_recall@1"] = (
        results["image_to_text_R@1"] + results["text_to_image_R@1"]
    ) / 2.0
    return results


@torch.no_grad()
def evaluate_retrieval(model, loader: DataLoader, device: str, ks: list[int] | None = None) -> Dict[str, float]:
    ks = ks or [1, 5, 10]
    image_embeds, text_embeds, product_ids = encode_dataset(model, loader, device)
    similarity = image_embeds @ text_embeds.T
    return compute_recall_at_k(similarity, product_ids, ks)