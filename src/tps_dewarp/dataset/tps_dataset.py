import json
import warnings
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.tps_dewarp.dataset.canvas_spatial import (
    CanvasSpatialSpec,
    DeltaTPSNormCanvasTransform,
    WarpedImageCanvasTransform,
    uint8_hw_to_float01_chw,
)


class TPSDataset(Dataset):
    """
    Датасет warped PNG + deltaTPS из metadata.json.

    Режимы:
    - **Legacy**: один ``transform`` на изображение после чтения с диска (как раньше).
    - **Согласованный канон**: ``spatial_spec`` + опционально ``photometric_transform``;
      ``deltaTPS`` пересчитывается в координаты выхода через ``DeltaTPSNormCanvasTransform``.

    Кеш ``lru_cache`` по умолчанию отключён (``maxsize=0``). При ``spatial_spec`` кешируется
    только сырое grayscale с диска; геометрия применяется на каждый ``__getitem__``.
    Параметр ``cache_images`` оставлен для совместимости вызовов и не влияет на размер кеша
    (используйте ``lru_cache_maxsize``).
    """

    def __init__(
        self,
        dataset_dir: str,
        transform: Callable | None = None,
        cache_images: bool = False,
        return_meta: bool = False,
        lru_cache_maxsize: int | None = 0,
        *,
        spatial_spec: CanvasSpatialSpec | None = None,
        image_canvas_transform: WarpedImageCanvasTransform | None = None,
        delta_canvas_transform: DeltaTPSNormCanvasTransform | None = None,
        photometric_transform: Callable | None = None,
        to_model_tensor: Callable[[np.ndarray], torch.Tensor] | None = None,
        grid_size: int = 9,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.return_meta = return_meta
        self.spatial_spec = spatial_spec
        self.photometric_transform = photometric_transform
        self.grid_size = int(grid_size)

        if cache_images is True and lru_cache_maxsize == 0:
            warnings.warn(
                "TPSDataset: cache_images=True but lru_cache_maxsize=0 disables caching; "
                "set lru_cache_maxsize>0 to cache raw images.",
                stacklevel=2,
            )

        if spatial_spec is not None and transform is not None:
            raise ValueError(
                "Use either legacy `transform` or `spatial_spec`, not both."
            )

        self._legacy_transform = transform
        self.lru_cache_maxsize = lru_cache_maxsize

        if spatial_spec is not None:
            self.image_canvas_transform = image_canvas_transform or WarpedImageCanvasTransform(
                fill=spatial_spec.fill
            )
            self.delta_canvas_transform = delta_canvas_transform or DeltaTPSNormCanvasTransform(
                grid_size=self.grid_size
            )
            self.to_model_tensor = to_model_tensor or uint8_hw_to_float01_chw

            @lru_cache(maxsize=lru_cache_maxsize)
            def cached_raw_loader(img_path: str):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(f"Cannot read image: {img_path}")
                return img

            self._load_raw = cached_raw_loader
        else:

            @lru_cache(maxsize=lru_cache_maxsize)
            def cached_loader(img_path: str):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(f"Cannot read image: {img_path}")
                if transform is not None:
                    img = transform(img)
                return img

            self._load_raw = None
            self._legacy_load = cached_loader
            self.image_canvas_transform = None
            self.delta_canvas_transform = None
            self.to_model_tensor = None

        meta_path = self.dataset_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError("metadata.json not found")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self._build_indices()

    def _build_indices(self):
        self.indices_by_difficulty: dict[str, list[int]] = {}

        for idx, item in enumerate(self.samples):
            diff = item["difficulty"]

            if diff not in self.indices_by_difficulty:
                self.indices_by_difficulty[diff] = []

            self.indices_by_difficulty[diff].append(idx)

    def get_indices_by_difficulty(self):
        return self.indices_by_difficulty

    def get_difficulties(self):
        return list(self.indices_by_difficulty.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        img_path = self.dataset_dir / item["warped"]
        path_str = str(img_path)

        delta = torch.tensor(item["deltaTPS"], dtype=torch.float32)
        difficulty = item["difficulty"]

        if self.spatial_spec is not None:
            raw = self._load_raw(path_str)
            h_disk, w_disk = raw.shape[:2]
            tensor01 = self.to_model_tensor(raw)
            ctx = self.spatial_spec.build(h_disk, w_disk)
            img = self.image_canvas_transform(tensor01, ctx)
            delta = self.delta_canvas_transform(delta, h_disk, w_disk, ctx)
            if self.photometric_transform is not None:
                img = self.photometric_transform(img)
        else:
            img = self._legacy_load(path_str)

        if self.return_meta:
            meta = {
                "original": item["original"],
                "warped": item["warped"],
                "is_identity": item["is_identity"],
            }
            if self.spatial_spec is not None:
                meta["canvas_h"] = self.spatial_spec.out_h
                meta["canvas_w"] = self.spatial_spec.out_w
            return img, delta, difficulty, meta

        return img, delta, difficulty
