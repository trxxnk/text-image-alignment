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
    loader_drop_last: bool
    loader_persistent_workers: bool
    loader_prefetch_factor: int

    model_name: str
    num_points: int
    use_coordconv: bool
    model_pretrained: bool
    tanh_output_scale: float

    loss_grid_size: int
    loss_lambda_smooth: float
    loss_difficulty_weights: dict[str, float]

    optimizer_name: str
    lr: float
    weight_decay: float

    scheduler_name: str

    epochs: int
    warmup_epochs: int
    use_amp: bool
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

        epochs = int(t["epochs"])
        warmup_epochs = int(t["warmup_epochs"])
        if warmup_epochs < 0 or warmup_epochs >= epochs:
            raise ValueError(f"warmup_epochs must satisfy 0 <= warmup_epochs < epochs, got {warmup_epochs}, {epochs}")

        tanh_scale = float(m["tanh_output_scale"])
        if tanh_scale <= 0:
            raise ValueError(f"tanh_output_scale must be > 0, got {tanh_scale}")

        dw_raw = lo["difficulty_weights"]
        if not isinstance(dw_raw, dict) or not dw_raw:
            raise ValueError("loss.difficulty_weights must be a non-empty mapping")
        difficulty_weights: dict[str, float] = {str(k): float(v) for k, v in dw_raw.items()}
        for k, v in difficulty_weights.items():
            if v <= 0:
                raise ValueError(f"loss.difficulty_weights[{k!r}] must be > 0, got {v}")

        prefetch = int(l["prefetch_factor"])
        if prefetch < 1:
            raise ValueError(f"loader.prefetch_factor must be >= 1, got {prefetch}")

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
            loader_drop_last=bool(l["drop_last"]),
            loader_persistent_workers=bool(l["persistent_workers"]),
            loader_prefetch_factor=prefetch,
            model_name=str(m["name"]),
            num_points=int(m["num_points"]),
            use_coordconv=bool(m["use_coordconv"]),
            model_pretrained=bool(m["pretrained"]),
            tanh_output_scale=tanh_scale,
            loss_grid_size=int(lo["grid_size"]),
            loss_lambda_smooth=float(lo["lambda_smooth"]),
            loss_difficulty_weights=difficulty_weights,
            optimizer_name=str(o["name"]),
            lr=float(o["lr"]),
            weight_decay=float(o["weight_decay"]),
            scheduler_name=str(sch["name"]),
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            use_amp=bool(t["use_amp"]),
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

        l = raw["loader"]
        for sub in ("drop_last", "persistent_workers", "prefetch_factor"):
            if sub not in l:
                raise KeyError(f"Missing loader.{sub} in {path}")

        m = raw["model"]
        for sub in ("pretrained", "tanh_output_scale"):
            if sub not in m:
                raise KeyError(f"Missing model.{sub} in {path}")

        lo = raw["loss"]
        if "difficulty_weights" not in lo:
            raise KeyError(f"Missing loss.difficulty_weights in {path}")

        t = raw["training"]
        for sub in ("warmup_epochs", "use_amp"):
            if sub not in t:
                raise KeyError(f"Missing training.{sub} in {path}")


def load_train_config(path: str | Path) -> TrainConfig:
    """Алиас для TrainConfig.from_yaml."""
    return TrainConfig.from_yaml(path)
