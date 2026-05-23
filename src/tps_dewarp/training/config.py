"""Загрузка YAML-конфига обучения."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    """Плоское представление nested YAML для Trainer и data_setup."""

    config_path: Path

    experiment_name: str
    run_name_prefix: str

    dataset_dir: str
    canvas_size: int
    spatial_mode: str
    letterbox_fill: float
    lru_cache_maxsize: int

    split_seed: int
    train_ratio: float
    val_ratio: float

    batch_size: int
    num_workers: int

    model_name: str
    num_points: int
    use_coordconv: bool

    loss_grid_size: int
    loss_lambda_smooth: float

    optimizer_name: str
    lr: float
    weight_decay: float

    scheduler_name: str

    epochs: int
    grad_clip_norm: float
    metric_for_best: str
    tqdm_enabled: bool
    l2_canvas_size: float

    checkpoint_dir: str

    mlflow_log_params: bool
    mlflow_log_config_artifact: bool
    mlflow_log_artifact_on_best: bool
    mlflow_log_final_model: bool

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Config not found: {p}")
        with open(p, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        cls._validate_keys(raw, p)

        d = raw["dataset"]
        s = raw["split"]
        l = raw["loader"]
        m = raw["model"]
        lo = raw["loss"]
        o = raw["optimizer"]
        sch = raw["scheduler"]
        t = raw["training"]
        c = raw["checkpoint"]
        ml = raw["mlflow"]

        metric = str(t["metric_for_best"])
        if metric not in ("val_loss", "val_l2_px"):
            raise ValueError(f"metric_for_best must be val_loss or val_l2_px, got {metric!r}")

        mode = str(d["spatial_mode"])
        if mode not in ("letterbox", "stretch"):
            raise ValueError(f"spatial_mode must be letterbox or stretch, got {mode!r}")

        return cls(
            config_path=p,
            experiment_name=str(raw["experiment"]["name"]),
            run_name_prefix=str(raw["run"]["name_prefix"]),
            dataset_dir=str(d["dir"]),
            canvas_size=int(d["canvas_size"]),
            spatial_mode=mode,
            letterbox_fill=float(d["letterbox_fill"]),
            lru_cache_maxsize=int(d["lru_cache_maxsize"]),
            split_seed=int(s["seed"]),
            train_ratio=float(s["train_ratio"]),
            val_ratio=float(s["val_ratio"]),
            batch_size=int(l["batch_size"]),
            num_workers=int(l["num_workers"]),
            model_name=str(m["name"]),
            num_points=int(m["num_points"]),
            use_coordconv=bool(m["use_coordconv"]),
            loss_grid_size=int(lo["grid_size"]),
            loss_lambda_smooth=float(lo["lambda_smooth"]),
            optimizer_name=str(o["name"]),
            lr=float(o["lr"]),
            weight_decay=float(o["weight_decay"]),
            scheduler_name=str(sch["name"]),
            epochs=int(t["epochs"]),
            grad_clip_norm=float(t["grad_clip_norm"]),
            metric_for_best=metric,
            tqdm_enabled=bool(t["tqdm"]),
            l2_canvas_size=float(t["l2_canvas_size"]),
            checkpoint_dir=str(c["dir"]),
            mlflow_log_params=bool(ml["log_params"]),
            mlflow_log_config_artifact=bool(ml["log_config_artifact"]),
            mlflow_log_artifact_on_best=bool(ml["log_artifact_on_best"]),
            mlflow_log_final_model=bool(ml["log_final_model"]),
        )

    @staticmethod
    def _validate_keys(raw: dict[str, Any], path: Path) -> None:
        required = [
            "experiment",
            "run",
            "dataset",
            "split",
            "loader",
            "model",
            "loss",
            "optimizer",
            "scheduler",
            "training",
            "checkpoint",
            "mlflow",
        ]
        for k in required:
            if k not in raw:
                raise KeyError(f"Missing key {k!r} in {path}")


def load_train_config(path: str | Path) -> TrainConfig:
    """Алиас для TrainConfig.from_yaml."""
    return TrainConfig.from_yaml(path)
