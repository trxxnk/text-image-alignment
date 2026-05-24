from __future__ import annotations

from collections import defaultdict

import torch


def compute_l2_px_per_sample(pred: torch.Tensor, target: torch.Tensor, canvas_size: float) -> torch.Tensor:
    """Средний L2 по точкам для каждого примера батча, в пикселях канона. Shape (B,)."""
    b = pred.shape[0]
    pred = pred.view(b, -1, 2)
    target = target.view(b, -1, 2)
    dist = torch.norm(pred - target, dim=2)
    return dist.mean(dim=1) * float(canvas_size)


def compute_l2_px_batch_mean(pred: torch.Tensor, target: torch.Tensor, canvas_size: float) -> float:
    """Среднее по батчу (как усреднение batch-mean в ноутбуке)."""
    return float(compute_l2_px_per_sample(pred, target, canvas_size).mean().item())


def aggregate_val_epoch_metrics(
    per_diff_sum: dict[str, float],
    per_diff_cnt: dict[str, int],
    global_l2_sum: float,
    global_n: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if global_n > 0:
        out["val_l2_px"] = global_l2_sum / global_n
    for d, s in per_diff_sum.items():
        c = per_diff_cnt.get(d, 0)
        if c > 0:
            out[f"val_l2_px_{d}"] = s / c
    return out


def quantile_metrics_1d(values: torch.Tensor, prefix: str) -> dict[str, float]:
    """p50, p95, max для 1D float тензора; ключи с префиксом."""
    out: dict[str, float] = {}
    if values.numel() == 0:
        return out
    v = values.detach().float().flatten()
    out[f"{prefix}_p50"] = float(torch.quantile(v, 0.5).item())
    out[f"{prefix}_p95"] = float(torch.quantile(v, 0.95).item())
    out[f"{prefix}_max"] = float(v.max().item())
    return out


def per_group_quantiles(
    per_group_values: dict[str, list[float]],
    value_prefix: str,
) -> dict[str, float]:
    """Для каждой группы: value_prefix_{group}_p95, _max (p50 опционально — добавим p95/max по плану)."""
    out: dict[str, float] = {}
    for group, vals in per_group_values.items():
        if not vals:
            continue
        t = torch.tensor(vals, dtype=torch.float32)
        out[f"{value_prefix}_{group}_p95"] = float(torch.quantile(t, 0.95).item())
        out[f"{value_prefix}_{group}_max"] = float(t.max().item())
    return out
