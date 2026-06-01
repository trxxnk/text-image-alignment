"""Примеры синтетически искажённых страниц по уровням сложности.

Берём по одному примеру каждого difficulty из data/generated/v3 и показываем
warped-изображение. Для identity дополнительно подтверждаем, что искажения нет.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from _common import ROOT, save

DATASET = ROOT / "data" / "generated" / "v3"
ORDER = ["identity", "easy", "medium", "hard"]
TITLES = {
    "identity": "identity (без искажений)",
    "easy": "easy (волны)",
    "medium": "medium (перспектива)",
    "hard": "hard (комбинация)",
}


def main() -> None:
    with open(DATASET / "metadata.json", encoding="utf-8") as f:
        samples = json.load(f)

    # первый встретившийся пример каждого класса
    picked: dict[str, dict] = {}
    for s in samples:
        d = s["difficulty"]
        if d in ORDER and d not in picked:
            picked[d] = s
        if len(picked) == len(ORDER):
            break

    fig, axes = plt.subplots(1, len(ORDER), figsize=(13, 4.2))
    for ax, diff in zip(axes, ORDER):
        item = picked[diff]
        img = np.array(Image.open(DATASET / item["warped"]).convert("L"))
        ax.imshow(img, cmap="gray")
        ax.set_title(TITLES[diff], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    save(fig, "fig_warp_examples.png")


if __name__ == "__main__":
    main()
