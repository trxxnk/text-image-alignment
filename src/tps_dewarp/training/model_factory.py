"""Фабрика моделей по TrainConfig."""

from __future__ import annotations

import torch.nn as nn

from src.models.tpsresnet18 import TPSResNet18
from src.tps_dewarp.training.config import TrainConfig


def build_model(cfg: TrainConfig) -> nn.Module:
    name = cfg.model_name
    if name == "TPSResNet18":
        return TPSResNet18(num_points=cfg.num_points, use_coordconv=cfg.use_coordconv)
    raise ValueError(f"Unknown model name: {name!r}")
