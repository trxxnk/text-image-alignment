"""
Согласованное приведение warped-изображения и deltaTPS к одному канвасу (letterbox / stretch).

Контекст SyncContext считается один раз из (H_in, W_in); картинка и контрольные точки
используют одно и то же аффинное отображение диск -> выход.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torchvision.transforms.functional as F

Mode = Literal["letterbox", "stretch"]


def build_base_grid(H: int, W: int, grid_size: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Базовая сетка в пикселях диска, порядок как в TPSDatasetGenerator:
    for y in ys: for x in xs -> (G*G, 2) с колонками [x, y].
    """
    xs = torch.linspace(0, W - 1, grid_size, device=device, dtype=dtype)
    ys = torch.linspace(0, H - 1, grid_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


@dataclass(frozen=True)
class SyncContext:
    """Параметры отображения диск -> канон out_h x out_w."""

    h_in: int
    w_in: int
    h_out: int
    w_out: int
    mode: Mode
    # letterbox
    scale: float | None = None
    pad_left: int = 0
    pad_top: int = 0
    new_h: int | None = None
    new_w: int | None = None
    # stretch
    sx: float | None = None
    sy: float | None = None

    def map_points(self, xy: torch.Tensor) -> torch.Tensor:
        """
        xy: (..., 2) — столбцы [x, y] в координатах входного изображения (диск).
        Возвращает (..., 2) в координатах выходного канваса.
        """
        x = xy[..., 0]
        y = xy[..., 1]
        if self.mode == "letterbox":
            assert self.scale is not None
            x_out = x * self.scale + self.pad_left
            y_out = y * self.scale + self.pad_top
        else:
            assert self.sx is not None and self.sy is not None
            x_out = x * self.sx
            y_out = y * self.sy
        return torch.stack([x_out, y_out], dim=-1)


class CanvasSpatialSpec:
    """Описание выходного канваса; build(H_in, W_in) даёт SyncContext для пары трансформеров."""

    def __init__(
        self,
        out_h: int,
        out_w: int,
        mode: Mode = "letterbox",
        fill: float = 0.0,
    ):
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.mode: Mode = mode
        self.fill = float(fill)

    def build(self, h_in: int, w_in: int) -> SyncContext:
        h_in, w_in = int(h_in), int(w_in)
        if self.mode == "letterbox":
            scale = min(self.out_h / h_in, self.out_w / w_in)
            new_h = int(h_in * scale)
            new_w = int(w_in * scale)
            pad_h = self.out_h - new_h
            pad_w = self.out_w - new_w
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            return SyncContext(
                h_in=h_in,
                w_in=w_in,
                h_out=self.out_h,
                w_out=self.out_w,
                mode="letterbox",
                scale=float(scale),
                pad_left=pad_left,
                pad_top=pad_top,
                new_h=new_h,
                new_w=new_w,
            )
        # stretch
        sx = (self.out_w - 1) / max(w_in - 1, 1)
        sy = (self.out_h - 1) / max(h_in - 1, 1)
        return SyncContext(
            h_in=h_in,
            w_in=w_in,
            h_out=self.out_h,
            w_out=self.out_w,
            mode="stretch",
            sx=float(sx),
            sy=float(sy),
        )


class WarpedImageCanvasTransform:
    """Геометрия изображения в float tensor (1, H_in, W_in) -> (1, H_out, W_out)."""

    def __call__(self, img: torch.Tensor, ctx: SyncContext) -> torch.Tensor:
        if ctx.mode == "letterbox":
            assert ctx.scale is not None and ctx.new_h is not None and ctx.new_w is not None
            _, h, w = img.shape
            if h != ctx.h_in or w != ctx.w_in:
                raise ValueError(f"img shape {(h, w)} != ctx {(ctx.h_in, ctx.w_in)}")
            img = F.resize(img, [ctx.new_h, ctx.new_w])
            pad_h = ctx.h_out - ctx.new_h
            pad_w = ctx.w_out - ctx.new_w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            return F.pad(
                img,
                [pad_left, pad_top, pad_right, pad_bottom],
                fill=self.fill,
            )
        _, h, w = img.shape
        if h != ctx.h_in or w != ctx.w_in:
            raise ValueError(f"img shape {(h, w)} != ctx {(ctx.h_in, ctx.w_in)}")
        return F.resize(img, [ctx.h_out, ctx.w_out])

    def __init__(self, fill: float = 0.0):
        self.fill = float(fill)


class DeltaTPSNormCanvasTransform:
    """delta_norm на диске -> delta_norm на каноне ctx (согласовано с WarpedImageCanvasTransform)."""

    def __init__(self, grid_size: int = 9):
        self.grid_size = int(grid_size)

    def __call__(
        self,
        delta_norm_disk: torch.Tensor,
        h_disk: int,
        w_disk: int,
        ctx: SyncContext,
    ) -> torch.Tensor:
        if delta_norm_disk.ndim != 2 or delta_norm_disk.shape[1] != 2:
            raise ValueError(f"delta_norm_disk expected (N, 2), got {delta_norm_disk.shape}")
        device = delta_norm_disk.device
        dtype = delta_norm_disk.dtype
        h_disk = int(h_disk)
        w_disk = int(w_disk)
        denom_x = max(w_disk - 1, 1)
        denom_y = max(h_disk - 1, 1)
        base = build_base_grid(h_disk, w_disk, self.grid_size, device=device, dtype=dtype)
        delta_px = delta_norm_disk * torch.tensor(
            [denom_x, denom_y], device=device, dtype=dtype
        )
        warped = base + delta_px
        base_out = ctx.map_points(base)
        warped_out = ctx.map_points(warped)
        out_dx = max(ctx.w_out - 1, 1)
        out_dy = max(ctx.h_out - 1, 1)
        delta_px_out = warped_out - base_out
        return delta_px_out / torch.tensor([out_dx, out_dy], device=device, dtype=dtype)


def uint8_hw_to_float01_chw(raw: np.ndarray) -> torch.Tensor:
    """Grayscale HW uint8 -> (1, H, W) float32 [0, 1]."""
    if raw.ndim != 2:
        raise ValueError(f"expected HW grayscale, got shape {raw.shape}")
    t = torch.from_numpy(raw).to(dtype=torch.float32).unsqueeze(0) / 255.0
    return t
