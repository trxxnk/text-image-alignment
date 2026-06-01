"""val_l2_px по уровням сложности (run 04_colab) по эпохам.

Наглядно показывает: identity почти ноль, а волны (easy) «заморожены» на ~9.9 px
все 40 эпох — модель не учится исправлять высокочастотные искажения.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _common import DIFF_COLORS, DIFF_LABELS, read_metric, save


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))

    for diff in ("easy", "hard", "medium", "identity"):
        steps, vals = read_metric(f"val_l2_px_{diff}")
        ax.plot(steps, vals, label=DIFF_LABELS[diff], color=DIFF_COLORS[diff], lw=1.9)

    steps, vals = read_metric("val_l2_px")
    ax.plot(steps, vals, label="все классы (global)", color="black", lw=2.2, ls="--")

    ax.set_xlabel("Эпоха")
    ax.set_ylabel("val L2, пиксели канваса 256")
    ax.set_title("Ошибка по уровням сложности в процессе обучения")
    ax.legend(loc="center right")
    ax.set_ylim(0, None)

    save(fig, "fig_l2_by_difficulty.png")


if __name__ == "__main__":
    main()
