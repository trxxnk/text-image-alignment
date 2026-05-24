"""TPSDataset + DataLoader из TrainConfig."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split

from torchvision.transforms import v2

from src.tps_dewarp.dataset import CanvasSpatialSpec, TPSDataset
from src.tps_dewarp.training.config import TrainConfig


def build_tps_dataloaders(
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    spec = CanvasSpatialSpec(
        cfg.canvas_size,
        cfg.canvas_size,
        mode=cfg.spatial_mode,  # type: ignore[arg-type]
        fill=cfg.letterbox_fill,
    )
    photometric = v2.Normalize(mean=[0.5], std=[0.5])
    dataset = TPSDataset(
        cfg.dataset_dir,
        spatial_spec=spec,
        photometric_transform=photometric,
        lru_cache_maxsize=cfg.lru_cache_maxsize,
    )

    n = len(dataset)
    train_n = int(cfg.train_ratio * n)
    val_n = int(cfg.val_ratio * n)
    test_n = n - train_n - val_n
    if min(train_n, val_n, test_n) <= 0:
        raise ValueError(f"Invalid split sizes: train={train_n}, val={val_n}, test={test_n}, N={n}")

    gen = torch.Generator().manual_seed(cfg.split_seed)
    train_data, val_data, test_data = random_split(
        dataset, [train_n, val_n, test_n], generator=gen
    )

    pin = device.type == "cuda"
    base_kw: dict = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )
    if cfg.num_workers > 0:
        base_kw["persistent_workers"] = cfg.loader_persistent_workers
        base_kw["prefetch_factor"] = cfg.loader_prefetch_factor

    train_loader = DataLoader(train_data, shuffle=True, drop_last=cfg.loader_drop_last, **base_kw)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **base_kw)
    test_loader = DataLoader(test_data, shuffle=False, drop_last=False, **base_kw)
    return train_loader, val_loader, test_loader
