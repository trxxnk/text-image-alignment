import numpy as np

from . import BaseTransform


class RandomSpineBend(BaseTransform):
    def __init__(self,
                 strength_range=(10, 40)):
        self.strength_range = strength_range

    def sample(self, rng, H, W):
        self.strength = rng.uniform(*self.strength_range)
        self.side = rng.choice(["left", "right"])

    def apply_points(self, pts, H, W):
        pts = pts.copy()
        y_norm = (pts[:, 1] / H) - 0.5

        if self.side == "left":
            x_dist = pts[:, 0]
        else:
            x_dist = W - pts[:, 0]

        pts[:, 0] += self.strength * (y_norm**2) * (x_dist / W)
        return pts
