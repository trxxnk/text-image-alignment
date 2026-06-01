"""Тривиальные бейзлайны для задачи регрессии deltaTPS.

Считаем, какую метрику ``val_l2_px`` дают предсказания, не использующие изображение:
  * ``predict-zero``  -- модель всегда выдаёт нулевое смещение (deltaTPS = 0);
  * ``predict-mean``  -- модель всегда выдаёт среднее смещение по обучающей выборке.

Если обученная ResNet18 показывает L2, близкий к этим бейзлайнам, значит она
не извлекла из картинки полезного сигнала, а метрика просто отражает среднюю
амплитуду деформаций. Это и есть проверка гипотезы о "схлопывании" модели.

Метрика воспроизводит src/tps_dewarp/training/metrics.compute_l2_px_per_sample
с учётом приведения deltaTPS к каноническому канвасу (letterbox 256), как в обучении.
Размеры изображений читаются из заголовка PNG (без полного декодирования).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATASET_DIR = Path("data/generated/v3")
CANVAS = 256          # l2_canvas_size / canvas_size в обучении
GRID = 9
SPLIT_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


def letterbox_scale(h: int, w: int, out: int = CANVAS) -> float:
    return min(out / h, out / w)


def canvas_delta_norm(delta_disk: np.ndarray, h: int, w: int, out: int = CANVAS) -> np.ndarray:
    """delta_norm (диск) -> delta_norm (канвас letterbox).

    letterbox -- равномерный масштаб, паддинг сокращается в разности (warped - base),
    поэтому delta_px_out = scale * delta_px_disk, а нормировка идёт на (out - 1).
    """
    scale = letterbox_scale(h, w, out)
    delta_px_disk = delta_disk * np.array([w - 1, h - 1], dtype=np.float64)
    delta_px_out = scale * delta_px_disk
    return delta_px_out / (out - 1)


def l2_px(delta_norm_a: np.ndarray, delta_norm_b: np.ndarray, out: int = CANVAS) -> float:
    """Средний по точкам L2, в пикселях канваса (как compute_l2_px_per_sample)."""
    dist = np.linalg.norm(delta_norm_a - delta_norm_b, axis=-1)  # (N_points,)
    return float(dist.mean() * out)


def main() -> None:
    meta_path = DATASET_DIR / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    n = len(samples)
    print(f"Всего примеров: {n}")

    # --- тот же сплит, что в data_setup.build_tps_dataloaders ---
    train_n = int(TRAIN_RATIO * n)
    val_n = int(VAL_RATIO * n)
    gen = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n, generator=gen).tolist()
    train_idx = perm[:train_n]
    val_idx = perm[train_n:train_n + val_n]
    print(f"train={len(train_idx)}  val={len(val_idx)}")

    # --- канвас-дельты по всем нужным индексам ---
    def load_canvas_delta(i: int) -> tuple[np.ndarray, str]:
        item = samples[i]
        delta = np.asarray(item["deltaTPS"], dtype=np.float64)  # (81, 2) норм. на диске
        with Image.open(DATASET_DIR / item["warped"]) as im:
            w, h = im.size  # PIL: (width, height), без декодирования пикселей
        return canvas_delta_norm(delta, h, w), item["difficulty"]

    # среднее смещение по train (predict-mean)
    train_sum = np.zeros((GRID * GRID, 2), dtype=np.float64)
    for k, i in enumerate(train_idx):
        d, _ = load_canvas_delta(i)
        train_sum += d
        if (k + 1) % 5000 == 0:
            print(f"  train обработано {k + 1}/{len(train_idx)}")
    mean_delta = train_sum / len(train_idx)

    zeros = np.zeros((GRID * GRID, 2), dtype=np.float64)

    # метрики на val
    zero_all, mean_all = [], []
    zero_by_diff: dict[str, list[float]] = defaultdict(list)
    for k, i in enumerate(val_idx):
        d, diff = load_canvas_delta(i)
        lz = l2_px(zeros, d)
        lm = l2_px(mean_delta, d)
        zero_all.append(lz)
        mean_all.append(lm)
        zero_by_diff[diff].append(lz)
        if (k + 1) % 1000 == 0:
            print(f"  val обработано {k + 1}/{len(val_idx)}")

    print("\n================ РЕЗУЛЬТАТЫ (val, L2 в px канваса 256) ================")
    print(f"predict-zero  global: {np.mean(zero_all):.3f}  "
          f"(p50={np.percentile(zero_all, 50):.3f}, "
          f"p95={np.percentile(zero_all, 95):.3f}, max={np.max(zero_all):.3f})")
    print(f"predict-mean  global: {np.mean(mean_all):.3f}")
    print("\npredict-zero по сложности:")
    for diff in ("identity", "easy", "medium", "hard"):
        if diff in zero_by_diff:
            vals = zero_by_diff[diff]
            print(f"  {diff:9s}: {np.mean(vals):.3f}  (n={len(vals)})")

    print("\nДля сравнения, обученная ResNet18 (run 04_colab):")
    print("  global=6.08  identity=0.46  medium=4.07  hard=6.26  easy=9.91")

    out = {
        "predict_zero": {
            "global": float(np.mean(zero_all)),
            "p50": float(np.percentile(zero_all, 50)),
            "p95": float(np.percentile(zero_all, 95)),
            "max": float(np.max(zero_all)),
            "by_difficulty": {d: float(np.mean(v)) for d, v in zero_by_diff.items()},
        },
        "predict_mean": {"global": float(np.mean(mean_all))},
        "model_04_colab": {
            "global": 6.083, "identity": 0.455, "medium": 4.072,
            "hard": 6.260, "easy": 9.914,
        },
        "n_val": len(val_idx),
        "n_train": len(train_idx),
    }
    res_path = Path(__file__).resolve().parent / "baseline_results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены: {res_path}")


if __name__ == "__main__":
    main()
