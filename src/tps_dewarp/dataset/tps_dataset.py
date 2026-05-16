import json
import cv2
import torch
import numpy as np
from pathlib import Path
from functools import lru_cache
from torch.utils.data import Dataset


class TPSDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 transform=None,
                 cache_images: bool = False,
                 return_meta: bool = False,
                 lru_cache_maxsize: int | None = 0):

        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.lru_cache_maxsize = lru_cache_maxsize

        @lru_cache(maxsize=lru_cache_maxsize)
        def cached_loader(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if transform is not None:
                img = transform(img)
            return img

        self.cached_loader = cached_loader

        self.return_meta = return_meta

        meta_path = self.dataset_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError("metadata.json not found")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        # ===== индекс для sampler =====
        self._build_indices()

    # ===== индексация по difficulty =====
    def _build_indices(self):
        self.indices_by_difficulty: dict[str, list[int]] = {}

        for idx, item in enumerate(self.samples):
            diff = item["difficulty"]

            if diff not in self.indices_by_difficulty:
                self.indices_by_difficulty[diff] = []

            self.indices_by_difficulty[diff].append(idx)

    # ===== API для sampler =====
    def get_indices_by_difficulty(self):
        return self.indices_by_difficulty

    def get_difficulties(self):
        return list(self.indices_by_difficulty.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # ===== load image =====
        img_path = self.dataset_dir / item["warped"]
        img = self.cached_loader(img_path)

        # ===== load target deltaTPS =====
        delta = torch.tensor(item["deltaTPS"], dtype=torch.float32)

        # ===== load difficulty =====
        difficulty = item["difficulty"]

        if self.return_meta:
            return img, delta, difficulty, {
                "original": item["original"],
                "warped": item["warped"],
                "is_identity": item["is_identity"]
            }

        return img, delta, difficulty
