import json
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class TPSDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 transform=None):
        """
        dataset_dir:
            папка с metadata.json и изображениями

        transform:
            callable(image) -> tensor
            используется ТОЛЬКО для входа в нейросеть
        """

        self.dataset_dir = Path(dataset_dir)
        self.transform = transform

        meta_path = self.dataset_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError("metadata.json not found")

        with open(meta_path, "r") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # --- load image ---
        img_path = Path(item["warped"])
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        # --- apply transform for NN ---
        if self.transform is not None:
            img = self.transform(img)
        else:
            raise RuntimeError("Transformation cannot be `None`!")

        # --- load deltaTPS ---
        delta = np.array(item["deltaTPS"], dtype=np.float32)  # (25, 2)
        delta = torch.from_numpy(delta)

        return img, delta
