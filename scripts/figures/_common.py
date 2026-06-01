"""Общие утилиты для скриптов построения графиков диплома.

Читаем CSV-выгрузки метрик MLflow (DagsHub) формата:
    "Run","Run ID","metric","step","timestamp","value"
и отдаём массивы (steps, values), отсортированные по step.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

# Корни проекта и данных
ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "материалы_диплом_текст" / "04_colab"
OUT_DIR = ROOT / "images" / "thesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Единый стиль: крупный шрифт, сетка, без «няшности» — под печать.
mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "figure.autolayout": True,
})

# Человекочитаемые подписи уровней сложности
DIFF_LABELS = {
    "identity": "identity (без искажений)",
    "easy": "easy (волны)",
    "medium": "medium (перспектива)",
    "hard": "hard (комбинация)",
}
DIFF_COLORS = {
    "identity": "#2ca02c",
    "easy": "#d62728",
    "medium": "#1f77b4",
    "hard": "#ff7f0e",
}


def read_metric(name: str) -> tuple[list[int], list[float]]:
    """Прочитать <name>.csv из папки метрик -> (steps, values)."""
    path = METRICS_DIR / f"{name}.csv"
    rows: list[tuple[int, float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((int(r["step"]), float(r["value"])))
    rows.sort(key=lambda t: t[0])
    steps = [s for s, _ in rows]
    vals = [v for _, v in rows]
    return steps, vals


def save(fig, name: str) -> Path:
    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight")
    print(f"saved: {out}")
    return out
