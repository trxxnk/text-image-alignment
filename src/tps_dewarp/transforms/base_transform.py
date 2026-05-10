import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List


class BaseTransform:
    def __init__(self):
        self.strength = 1.0

    def set_strength(self, strength):
        self.strength = strength

    def sample(self, rng: np.random.Generator, H: int, W: int):
        """Сэмплирование случайных параметров"""
        pass

    def apply_points(self, pts: np.ndarray, H: int, W: int) -> np.ndarray:
        raise NotImplementedError
