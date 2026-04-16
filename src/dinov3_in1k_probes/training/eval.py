"""Classification evaluation on pre-extracted features (all in GPU memory)."""

from typing import Callable, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor


class ClfResult(NamedTuple):
    loss: float
    top1: float
    top5: float
    predictions: Tensor  # (N,) int64


def evaluate(
    model: nn.Module,
    *,
    features: Tensor,
    labels: Tensor,
    batch_size: int,
    loss_fn: Callable,
) -> ClfResult:
    model.eval()
    device = features.device
    N = len(features)
    total_loss = torch.tensor(0.0, device=device)
    total_top1 = torch.tensor(0, device=device)
    total_top5 = torch.tensor(0, device=device)
    predictions = torch.empty(N, dtype=torch.long, device=device)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            feat = features[i : i + batch_size]
            lbl = labels[i : i + batch_size].long()

            logits = model(feat)
            total_loss += loss_fn(logits, lbl, reduction="mean") * len(feat)

            preds = logits.argmax(dim=1)
            predictions[i : i + len(feat)] = preds
            total_top1 += (preds == lbl).sum()
            total_top5 += (logits.topk(5, dim=1).indices == lbl.unsqueeze(1)).any(1).sum()

    return ClfResult(
        loss=(total_loss / N).item(),
        top1=(total_top1 / N).item(),
        top5=(total_top5 / N).item(),
        predictions=predictions,
    )
