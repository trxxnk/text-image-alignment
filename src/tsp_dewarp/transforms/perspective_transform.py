import numpy as np
import cv2

from . import BaseTransform


class RandomPerspectiveTransform(BaseTransform):
    def __init__(self,
                 distortion_scale_range=(0.03, 0.10)):
        """
        distortion_scale ~ доля от размера изображения
        0.03–0.10 = умеренная перспектива (реалистично)
        """
        self.distortion_scale_range = distortion_scale_range

    def sample(self, rng, H, W):
        scale = rng.uniform(*self.distortion_scale_range)

        dx = scale * W
        dy = scale * H

        # исходные углы
        self.src = np.array([
            [0, 0],
            [W - 1, 0],
            [W - 1, H - 1],
            [0, H - 1]
        ], dtype=np.float32)

        # искажённые углы
        self.dst = self.src + rng.uniform(
            low=[-dx, -dy],
            high=[dx, dy],
            size=(4, 2)
        ).astype(np.float32)

        # матрица гомографии
        self.H = cv2.getPerspectiveTransform(self.src, self.dst)

    def apply_points(self, pts, H, W):
        """
        pts: (N, 2) в пикселях
        """
        pts = pts.copy()

        pts_h = np.concatenate(
            [pts, np.ones((pts.shape[0], 1))],
            axis=1
        )  # (N, 3)

        warped = (self.H @ pts_h.T).T
        warped = warped[:, :2] / warped[:, 2:3]

        return warped
