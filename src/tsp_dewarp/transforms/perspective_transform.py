import numpy as np
import cv2

from . import BaseTransform


class RandomPerspectiveTransform(BaseTransform):
    def __init__(self,
                 distortion_scale_range=(0.03, 0.10)):
        """
        distortion_scale — доля от размера изображения
        """
        super().__init__()

        self.distortion_scale_range = distortion_scale_range
        self.strength = 1.0

    def set_strength(self, strength: float):
        self.strength = strength

    def sample(self, rng, H, W):
        # --- нормализованный scale ---
        scale = rng.uniform(*self.distortion_scale_range) * self.strength

        dx = scale * W
        dy = scale * H

        self.src = np.array([
            [0, 0],
            [W - 1, 0],
            [W - 1, H - 1],
            [0, H - 1]
        ], dtype=np.float32)

        self.dst = self.src + rng.uniform(
            low=[-dx, -dy],
            high=[dx, dy],
            size=(4, 2)
        ).astype(np.float32)

        # --- матрица гомографии ---
        self.H_mat = cv2.getPerspectiveTransform(self.src, self.dst)

    def apply_points(self, pts, H, W):
        pts = pts.astype(np.float32, copy=True)

        pts_h = np.concatenate(
            [pts, np.ones((pts.shape[0], 1), dtype=np.float32)],
            axis=1
        )

        warped = (self.H_mat @ pts_h.T).T

        # защита от деления на 0
        z = warped[:, 2:3]
        z[z == 0] = 1e-6

        warped = warped[:, :2] / z

        return warped
