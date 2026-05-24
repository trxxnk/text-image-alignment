"""Цикл обучения: train/val, чекпойнты, MLflow."""

from __future__ import annotations

import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch import amp as torch_amp
from torch import nn, optim
from tqdm import tqdm

from src.tps_dewarp.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from src.tps_dewarp.training.config import TrainConfig
from src.tps_dewarp.training.losses import TPSLossRegularization
from src.tps_dewarp.training.metrics import (
    aggregate_val_epoch_metrics,
    compute_l2_px_per_sample,
    per_group_quantiles,
    quantile_metrics_1d,
)


def _difficulty_list(diffs: Any, batch_size: int) -> list[str]:
    if isinstance(diffs, str):
        return [diffs] * batch_size
    dl = list(diffs)
    if len(dl) != batch_size:
        return [str(diffs)] * batch_size
    return [str(x) for x in dl]


def _optimizer_state_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _build_scheduler(cfg: TrainConfig, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
    if cfg.scheduler_name != "CosineAnnealingLR":
        raise ValueError(f"Unsupported scheduler: {cfg.scheduler_name!r}")
    if cfg.warmup_epochs > 0:
        w = cfg.warmup_epochs
        cos_epochs = cfg.epochs - w
        if cos_epochs < 1:
            raise ValueError("epochs - warmup_epochs must be >= 1 when warmup_epochs > 0")
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=w,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cos_epochs)
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[w],
        )
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)


def _flatten_config_for_mlflow(cfg: TrainConfig) -> dict[str, str]:
    from dataclasses import asdict

    flat: dict[str, str] = {}
    for k, v in asdict(cfg).items():
        if k == "config_path":
            flat["config_path"] = str(v)
        elif k == "loss_difficulty_weights":
            flat[k] = str(dict(v))
        else:
            flat[k] = str(v)
    return flat


def _mlflow_params_dict(cfg: TrainConfig) -> dict[str, str]:
    """Все параметры конфига + короткие имена под колонки DagsHub/MLflow (model, img_size, grid_size)."""
    flat = _flatten_config_for_mlflow(cfg)
    # Как в старых ноутбуках: колонки model / img_size / grid_size
    aliases: dict[str, str] = {
        "model": f"{cfg.model_name.lower()}_resize{cfg.canvas_size}",
        "img_size": str(cfg.canvas_size),
        "grid_size": str(cfg.loss_grid_size),
        "num_points": str(cfg.num_points),
        "batch_size": str(cfg.batch_size),
        "epochs": str(cfg.epochs),
        "warmup_epochs": str(cfg.warmup_epochs),
        "lr": str(cfg.lr),
        "weight_decay": str(cfg.weight_decay),
        "optimizer": cfg.optimizer_name,
        "scheduler": cfg.scheduler_name,
        "metric_for_best": cfg.metric_for_best,
        "spatial_mode": cfg.spatial_mode,
        "dataset_dir": cfg.dataset_dir,
        "pretrained": str(cfg.model_pretrained),
        "use_coordconv": str(cfg.use_coordconv),
        "use_amp": str(cfg.use_amp),
        "tanh_output_scale": str(cfg.tanh_output_scale),
        "lambda_smooth": str(cfg.loss_lambda_smooth),
        "grad_clip_norm": str(cfg.grad_clip_norm),
        "l2_canvas_size": str(cfg.l2_canvas_size),
        "split_seed": str(cfg.split_seed),
        "train_ratio": str(cfg.train_ratio),
        "val_ratio": str(cfg.val_ratio),
        "num_workers": str(cfg.num_workers),
        "loader_drop_last": str(cfg.loader_drop_last),
    }
    flat.update(aliases)
    return flat


def _mlflow_log_params_batched(params: dict[str, str], batch_size: int = 90) -> None:
    """MLflow ограничивает число параметров за один log_params (~100)."""
    items = [(k, str(v)[:500]) for k, v in params.items()]
    for i in range(0, len(items), batch_size):
        chunk = dict(items[i : i + batch_size])
        mlflow.log_params(chunk)


def _mlflow_set_config_tags(cfg: TrainConfig) -> None:
    """Теги run — часто отображаются в UI даже если колонки завязаны на tags."""
    if mlflow.active_run() is None:
        return
    mlflow.set_tags(
        {
            "model": f"{cfg.model_name.lower()}_resize{cfg.canvas_size}",
            "img_size": str(cfg.canvas_size),
            "grid_size": str(cfg.loss_grid_size),
        }
    )


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.loss_fn = TPSLossRegularization(
            grid_size=cfg.loss_grid_size,
            lambda_smooth=cfg.loss_lambda_smooth,
        ).to(device)

        if cfg.optimizer_name != "AdamW":
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer_name!r}")
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.scheduler = _build_scheduler(cfg, self.optimizer)

        self._use_amp = bool(cfg.use_amp) and device.type == "cuda"
        self.scaler = torch_amp.GradScaler("cuda") if self._use_amp else None

        self._opened_new_mlflow_run = False

    def _difficulty_weight_tensor(self, diffs_list: list[str], device: torch.device) -> torch.Tensor:
        w = [self.cfg.loss_difficulty_weights.get(d, 1.0) for d in diffs_list]
        return torch.tensor(w, device=device, dtype=torch.float32)

    def _scaler_payload(self) -> dict[str, Any] | None:
        if self.scaler is None:
            return None
        return self.scaler.state_dict()

    def _ensure_mlflow_run(self, resume_run_id: str | None) -> None:
        if mlflow.active_run() is not None:
            rid = mlflow.active_run().info.run_id
            if resume_run_id and rid != resume_run_id:
                mlflow.end_run()
            else:
                return

        mlflow.set_experiment(self.cfg.experiment_name)
        if resume_run_id:
            mlflow.start_run(run_id=resume_run_id)
            self._opened_new_mlflow_run = False
            return

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.cfg.run_name_prefix}_{ts}"
        mlflow.start_run(run_name=run_name)
        self._opened_new_mlflow_run = True

    def _log_mlflow_params_once(self) -> None:
        if not self.cfg.mlflow_log_params:
            return
        if mlflow.active_run() is None:
            return
        params = _mlflow_params_dict(self.cfg)
        _mlflow_log_params_batched(params)
        if self.cfg.mlflow_log_config_artifact:
            mlflow.log_artifact(str(self.cfg.config_path), artifact_path="config")

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        weighted_loss_num = 0.0
        weighted_loss_den = 0.0
        train_l2_sum = 0.0
        train_l2_n = 0
        grad_norm_sum = 0.0
        grad_norm_batches = 0
        n_batches = 0

        iterator = self.train_loader
        if self.cfg.tqdm_enabled:
            iterator = tqdm(self.train_loader, leave=False, desc="train")

        for x, targets, diffs in iterator:
            x = x.to(self.device)
            targets = targets.to(self.device)
            b = x.size(0)
            dl = _difficulty_list(diffs, b)
            w = self._difficulty_weight_tensor(dl, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch_amp.autocast("cuda", dtype=torch.float16, enabled=self._use_amp):
                pred = self.model(x)
                loss_vec = self.loss_fn.per_sample_loss(pred, targets)
                loss = (loss_vec * w).sum() / w.sum().clamp(min=1e-8)

            if torch.isnan(loss).any():
                warnings.warn("NaN loss in training batch; skipping backward/step", UserWarning)
                continue

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                gn = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.grad_clip_norm,
                    ).item()
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                gn = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.grad_clip_norm,
                    ).item()
                )
                self.optimizer.step()

            grad_norm_sum += gn
            grad_norm_batches += 1

            weighted_loss_num += float((loss_vec.detach() * w).sum().item())
            weighted_loss_den += float(w.sum().item())

            px = compute_l2_px_per_sample(pred.detach(), targets.detach(), self.cfg.l2_canvas_size)
            train_l2_sum += float(px.sum().item())
            train_l2_n += int(px.numel())
            n_batches += 1

        if n_batches == 0:
            warnings.warn("No training batches processed in epoch", UserWarning)
            return {
                "train_loss": 0.0,
                "train_l2_px": 0.0,
                "grad_norm_mean": 0.0,
                "train_batches": 0.0,
                "train_samples": 0.0,
            }

        return {
            "train_loss": weighted_loss_num / max(weighted_loss_den, 1e-8),
            "train_l2_px": train_l2_sum / max(train_l2_n, 1),
            "grad_norm_mean": grad_norm_sum / max(grad_norm_batches, 1),
            "train_batches": float(n_batches),
            "train_samples": float(train_l2_n),
        }

    def validate(self) -> dict[str, float]:
        self.model.eval()
        all_loss: list[float] = []
        all_l2: list[float] = []
        per_diff_loss: dict[str, list[float]] = defaultdict(list)
        per_diff_l2: dict[str, list[float]] = defaultdict(list)

        iterator = self.val_loader
        if self.cfg.tqdm_enabled:
            iterator = tqdm(self.val_loader, leave=False, desc="val")

        with torch.no_grad():
            for x, targets, diffs in iterator:
                x = x.to(self.device)
                targets = targets.to(self.device)
                b = x.size(0)
                pred = self.model(x)
                ls = self.loss_fn.per_sample_loss(pred, targets)
                px = compute_l2_px_per_sample(pred, targets, self.cfg.l2_canvas_size)
                dl = _difficulty_list(diffs, b)
                for i in range(b):
                    d = dl[i]
                    li = float(ls[i].item())
                    pi = float(px[i].item())
                    all_loss.append(li)
                    all_l2.append(pi)
                    per_diff_loss[d].append(li)
                    per_diff_l2[d].append(pi)

        metrics: dict[str, float] = {}
        n = len(all_loss)
        if n == 0:
            warnings.warn("No validation samples processed", UserWarning)
            metrics["val_loss"] = 0.0
            return metrics

        metrics["val_loss"] = sum(all_loss) / n

        per_diff_sum = {d: sum(v) for d, v in per_diff_l2.items()}
        per_diff_cnt = {d: len(v) for d, v in per_diff_l2.items()}
        global_l2_sum = sum(all_l2)
        global_l2_n = len(all_l2)
        metrics.update(aggregate_val_epoch_metrics(per_diff_sum, per_diff_cnt, global_l2_sum, global_l2_n))

        l2_t = torch.tensor(all_l2, dtype=torch.float32)
        metrics.update(quantile_metrics_1d(l2_t, "val_l2_px"))

        for d, lst in per_diff_loss.items():
            if lst:
                metrics[f"val_loss_{d}"] = sum(lst) / len(lst)

        metrics.update(per_group_quantiles(per_diff_l2, "val_l2_px"))

        return metrics

    def _current_score(self, metrics: dict[str, float]) -> float:
        key = self.cfg.metric_for_best
        if key not in metrics:
            raise KeyError(f"metric_for_best={key!r} not in metrics keys: {list(metrics)}")
        return float(metrics[key])

    def _save_last(
        self,
        epoch: int,
        best_metric_value: float,
        mlflow_run_id: str | None,
    ) -> None:
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_training_checkpoint(
            ckpt_dir / "last.pt",
            epoch=epoch,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            best_metric_value=best_metric_value,
            metric_for_best=self.cfg.metric_for_best,
            mlflow_run_id=mlflow_run_id,
            config_path=str(self.cfg.config_path),
            scaler_state_dict=self._scaler_payload(),
        )

    def fit(self, resume_from: str | Path | None = None) -> None:
        start_epoch = 0
        best_metric = float("inf")
        resume_run_id: str | None = None

        if resume_from is not None:
            ckpt = load_training_checkpoint(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            _optimizer_state_to_device(self.optimizer, self.device)
            start_epoch = int(ckpt["epoch"]) + 1
            best_metric = float(ckpt["best_metric_value"])
            resume_run_id = ckpt.get("mlflow_run_id")
            sd = ckpt.get("scaler_state_dict")
            if self.scaler is not None and isinstance(sd, dict):
                self.scaler.load_state_dict(sd)

        self._ensure_mlflow_run(resume_run_id=resume_run_id)
        active = mlflow.active_run()
        mlflow_run_id = active.info.run_id if active else None

        if mlflow.active_run() is not None:
            _mlflow_set_config_tags(self.cfg)

        if resume_from is None:
            self._log_mlflow_params_once()

        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            for epoch in range(start_epoch, self.cfg.epochs):
                t0 = time.perf_counter()
                train_stats = self.train_epoch()
                metrics = self.validate()
                self.scheduler.step()
                elapsed = time.perf_counter() - t0
                current_lr = self.optimizer.param_groups[0]["lr"]

                n_train = int(train_stats.get("train_samples", 0))
                n_val = len(self.val_loader.dataset)
                total_samples = float(n_train + n_val)
                metrics.update(
                    {
                        "train_loss": float(train_stats["train_loss"]),
                        "train_l2_px": float(train_stats["train_l2_px"]),
                        "grad_norm_mean": float(train_stats["grad_norm_mean"]),
                        "lr": current_lr,
                        "epoch_time_sec": float(elapsed),
                        "samples_per_sec": float(total_samples / elapsed) if elapsed > 0 else 0.0,
                    }
                )

                print(
                    f"Epoch {epoch + 1}/{self.cfg.epochs} | "
                    f"train_loss={metrics['train_loss']:.4f} | train_l2_px={metrics['train_l2_px']:.2f} | "
                    f"val_loss={metrics['val_loss']:.4f} | val_l2_px={metrics.get('val_l2_px', float('nan')):.2f} | "
                    f"p95={metrics.get('val_l2_px_p95', float('nan')):.2f} | "
                    f"gn={metrics['grad_norm_mean']:.3f} | lr={current_lr:.2e} | "
                    f"{metrics['samples_per_sec']:.1f} samp/s"
                )

                if mlflow.active_run() is not None:
                    log_payload = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                    mlflow.log_metrics(log_payload, step=epoch)

                score = self._current_score(metrics)
                improved = score < best_metric
                if improved:
                    best_metric = score
                    save_training_checkpoint(
                        ckpt_dir / "best.pt",
                        epoch=epoch,
                        model_state_dict=self.model.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        scheduler_state_dict=self.scheduler.state_dict(),
                        best_metric_value=best_metric,
                        metric_for_best=self.cfg.metric_for_best,
                        mlflow_run_id=mlflow_run_id,
                        config_path=str(self.cfg.config_path),
                        scaler_state_dict=self._scaler_payload(),
                    )
                    if self.cfg.mlflow_log_artifact_on_best and mlflow.active_run() is not None:
                        mlflow.log_artifact(str(ckpt_dir / "best.pt"), artifact_path="checkpoints")

                self._save_last(epoch, best_metric, mlflow_run_id)

            save_training_checkpoint(
                ckpt_dir / "final.pt",
                epoch=self.cfg.epochs - 1,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                best_metric_value=best_metric,
                metric_for_best=self.cfg.metric_for_best,
                mlflow_run_id=mlflow_run_id,
                config_path=str(self.cfg.config_path),
                scaler_state_dict=self._scaler_payload(),
            )
            if self.cfg.mlflow_log_final_model and mlflow.active_run() is not None:
                mlflow.pytorch.log_model(
                    self.model,
                    artifact_path="final_model",
                    registered_model_name=None,
                )
        finally:
            if self._opened_new_mlflow_run and mlflow.active_run() is not None:
                mlflow.end_run()
