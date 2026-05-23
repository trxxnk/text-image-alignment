"""Сохранение и загрузка чекпойнтов (resume, MLflow run_id)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


CHECKPOINT_VERSION = 1


def save_training_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    scheduler_state_dict: dict[str, Any],
    best_metric_value: float,
    metric_for_best: str,
    mlflow_run_id: str | None,
    config_path: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "best_metric_value": float(best_metric_value),
        "metric_for_best": str(metric_for_best),
        "mlflow_run_id": mlflow_run_id,
        "config_path": config_path,
    }
    torch.save(payload, path)


def load_training_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
