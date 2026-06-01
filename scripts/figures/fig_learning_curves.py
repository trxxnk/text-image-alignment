"""Кривые обучения run 04_colab: loss (train/val) и метрика val_l2_px по эпохам.

Слева — функция потерь в логарифмическом масштабе (так видно и быстрый спад
на 1-й эпохе, и дальнейшее «плато»). Справа — целевая метрика val_l2_px (px).
Дополнительно тонкой линией показано расписание learning rate.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _common import read_metric, save


def main() -> None:
    e_tr, train_loss = read_metric("train_loss")
    e_vl, val_loss = read_metric("val_loss")
    e_l2, val_l2 = read_metric("val_l2_px")
    e_lr, lr = read_metric("lr")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- (а) функции потерь, log-ось ---
    ax1.plot(e_tr, train_loss, label="train loss", color="#1f77b4", lw=1.8)
    ax1.plot(e_vl, val_loss, label="val loss", color="#d62728", lw=1.8)
    ax1.set_yscale("log")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Loss (SmoothL1 + reg), log")
    ax1.set_title("(а) Функция потерь")
    ax1.legend()

    # --- (б) val_l2_px + learning rate ---
    ax2.plot(e_l2, val_l2, label="val L2, px", color="#2ca02c", lw=2.0, marker="o", ms=3)
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("val L2, пиксели канваса 256")
    ax2.set_title("(б) Целевая метрика и LR")
    ax2.set_ylim(0, max(val_l2) * 1.15)

    ax2r = ax2.twinx()
    ax2r.plot(e_lr, lr, label="learning rate", color="#7f7f7f", lw=1.2, ls="--")
    ax2r.set_ylabel("learning rate")
    ax2r.grid(False)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    save(fig, "fig_learning_curves.png")


if __name__ == "__main__":
    main()
