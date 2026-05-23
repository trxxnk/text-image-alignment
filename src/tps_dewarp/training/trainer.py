"""Цикл обучения: train/val, чекпойнты, MLflow."""

from __future__ import annotations

import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch import nn, optim
from tqdm import tqdm

from src.tps_dewarp.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from src.tps_dewarp.training.config import TrainConfig
from src.tps_dewarp.training.losses import TPSLossRegularization
from src.tps_dewarp.training.metrics import aggregate_val_epoch_metrics, compute_l2_px_per_sample


def _flatten_config_for_mlflow(cfg: TrainConfig) -> dict[str, str]:
    from dataclasses import asdict

    flat: dict[str, str] = {}
    for k, v in asdict(cfg).items():
        if k == "config_path":
            flat["config_path"] = str(v)
        else:
            flat[k] = str(v)
    return flat


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

        if cfg.scheduler_name != "CosineAnnealingLR":
            raise ValueError(f"Unsupported scheduler: {cfg.scheduler_name!r}")
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.epochs,
        )

        self._opened_new_mlflow_run = False

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
        params = _flatten_config_for_mlflow(self.cfg)
        mlflow.log_params(params)
        if self.cfg.mlflow_log_config_artifact:
            mlflow.log_artifact(str(self.cfg.config_path), artifact_path="config")

    def train_epoch(self) -> float:
        self.model.train()
        loss_sum = 0.0
        n_batches = 0
        iterator = self.train_loader
        if self.cfg.tqdm_enabled:
            iterator = tqdm(self.train_loader, leave=False, desc="train")

        for x, targets, _ in iterator:
            x = x.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(x)
            loss = self.loss_fn(pred, targets)
            if torch.isnan(loss).any():
                warnings.warn("NaN loss in training batch; skipping backward/step", UserWarning)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()
            loss_sum += float(loss.item())
            n_batches += 1

        if n_batches == 0:
            warnings.warn("No training batches processed in epoch", UserWarning)
            return 0.0
        return loss_sum / n_batches

    def validate(self) -> dict[str, float]:
        self.model.eval()
        loss_sum_weighted = 0.0
        n_samples = 0
        global_l2_sum = 0.0
        global_l2_n = 0
        per_diff_sum: dict[str, float] = defaultdict(float)
        per_diff_cnt: dict[str, int] = defaultdict(int)

        iterator = self.val_loader
        if self.cfg.tqdm_enabled:
            iterator = tqdm(self.val_loader, leave=False, desc="val")

        with torch.no_grad():
            for x, targets, diffs in iterator:
                x = x.to(self.device)
                targets = targets.to(self.device)
                b = x.size(0)
                pred = self.model(x)
                loss = self.loss_fn(pred, targets)
                loss_sum_weighted += float(loss.item()) * b
                n_samples += b

                px = compute_l2_px_per_sample(pred, targets, self.cfg.l2_canvas_size)
                if isinstance(diffs, str):
                    diffs_list = [diffs] * b
                else:
                    diffs_list = list(diffs)
                if len(diffs_list) != b:
                    diffs_list = [str(diffs)] * b

                for i in range(b):
                    d = str(diffs_list[i])
                    per_diff_sum[d] += float(px[i].item())
                    per_diff_cnt[d] += 1
                global_l2_sum += float(px.sum().item())
                global_l2_n += int(px.numel())

        val_loss = loss_sum_weighted / max(n_samples, 1)
        extra = aggregate_val_epoch_metrics(per_diff_sum, per_diff_cnt, global_l2_sum, global_l2_n)
        extra["val_loss"] = val_loss
        return extra

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
            start_epoch = int(ckpt["epoch"]) + 1
            best_metric = float(ckpt["best_metric_value"])
            resume_run_id = ckpt.get("mlflow_run_id")

        self._ensure_mlflow_run(resume_run_id=resume_run_id)
        active = mlflow.active_run()
        mlflow_run_id = active.info.run_id if active else None

        if resume_from is None:
            self._log_mlflow_params_once()

        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            for epoch in range(start_epoch, self.cfg.epochs):
                train_loss = self.train_epoch()
                metrics = self.validate()
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

                metrics["train_loss"] = train_loss
                metrics["lr"] = current_lr

                print(
                    f"Epoch {epoch + 1}/{self.cfg.epochs} | "
                    f"train={train_loss:.4f} | val_loss={metrics['val_loss']:.4f} | "
                    f"val_l2_px={metrics.get('val_l2_px', float('nan')):.2f} | lr={current_lr:.2e}"
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
