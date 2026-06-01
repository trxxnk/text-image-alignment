"""Фабрика моделей по TrainConfig."""

from __future__ import annotations

import torch.nn as nn

from src.models.tpsresnet18 import TPSResNet18
from src.models.unet_tps import ResUNetTPS, UNetTPS
from src.tps_dewarp.training.config import TrainConfig

_MODELS = {
    "TPSResNet18": TPSResNet18,
    "UNetTPS": UNetTPS,
    "ResUNetTPS": ResUNetTPS,
}


def build_model(cfg: TrainConfig) -> nn.Module:
    name = cfg.model_name
    if name not in _MODELS:
        raise ValueError(f"Unknown model name: {name!r}. Available: {sorted(_MODELS)}")
    return _MODELS[name](
        num_points=cfg.num_points,
        use_coordconv=cfg.use_coordconv,
        pretrained=cfg.model_pretrained,
        output_scale=cfg.tanh_output_scale,
    )
