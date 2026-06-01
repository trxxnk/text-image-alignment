"""Сравнение обученной ResNet18 с тривиальными бейзлайнами (predict-zero / predict-mean).

Главный вывод работы в одной картинке: обученная сеть практически НЕ отличается
от предсказания нулевого смещения. Данные берутся из scripts/baseline_results.json
(сгенерирован scripts/baseline_predict_zero.py). Если файла нет — используются
значения, посчитанные ранее на val (33428 примеров, seed 42).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import save

FALLBACK = {
    "predict_zero": {
        "global": 6.011,
        "by_difficulty": {"identity": 0.0, "easy": 9.902, "medium": 4.167, "hard": 6.300},
    },
    "predict_mean": {"global": 5.967},
    "model_04_colab": {"global": 6.083, "identity": 0.455, "medium": 4.072,
                       "hard": 6.260, "easy": 9.914},
}


def load_results() -> dict:
    p = Path(__file__).resolve().parents[1] / "baseline_results.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return FALLBACK


def main() -> None:
    res = load_results()
    pz = res["predict_zero"]
    model = res["model_04_colab"]

    groups = ["global", "identity", "medium", "hard", "easy"]
    labels = ["global", "identity", "medium\n(персп.)", "hard\n(комбо)", "easy\n(волны)"]

    model_vals = [model["global"], model["identity"], model["medium"],
                  model["hard"], model["easy"]]
    zero_vals = [pz["global"], pz["by_difficulty"]["identity"], pz["by_difficulty"]["medium"],
                 pz["by_difficulty"]["hard"], pz["by_difficulty"]["easy"]]

    x = np.arange(len(groups))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.6))
    b1 = ax.bar(x - w / 2, model_vals, w, label="ResNet18 (обучена)", color="#1f77b4")
    b2 = ax.bar(x + w / 2, zero_vals, w, label="predict-zero (бейзлайн)", color="#bbbbbb")

    ax.axhline(res["predict_mean"]["global"], color="#d62728", ls=":", lw=1.5,
               label=f"predict-mean global = {res['predict_mean']['global']:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("val L2, пиксели канваса 256")
    ax.set_title("Обученная модель против тривиальных бейзлайнов")
    ax.legend()
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=9)
    ax.set_ylim(0, max(max(model_vals), max(zero_vals)) * 1.18)

    save(fig, "fig_baseline_comparison.png")


if __name__ == "__main__":
    main()
