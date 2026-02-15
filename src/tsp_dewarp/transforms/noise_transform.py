import numpy as np

from . import BaseTransform


class RandomNoiseTransform(BaseTransform):
    def __init__(self,
                 sigma_range=(3.0, 10.0)):
        """
        sigma — стандартное отклонение шума в пикселях
        """
        self.sigma_range = sigma_range

    def sample(self, rng, H, W):
        self.sigma = rng.uniform(*self.sigma_range)
        self.rng = rng

    def apply_points(self, pts, H, W):
        noise = self.rng.normal(0, self.sigma, size=pts.shape)
        return pts + noise
