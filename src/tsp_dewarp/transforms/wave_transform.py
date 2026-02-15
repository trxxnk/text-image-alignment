import numpy as np

from . import BaseTransform

class RandomWaveTransform(BaseTransform):
    def __init__(self,
                 amp_x_range=(5, 20),
                 amp_y_range=(5, 20),
                 freq_range=(0.5, 2.0)):
        self.amp_x_range = amp_x_range
        self.amp_y_range = amp_y_range
        self.freq_range = freq_range

    def sample(self, rng, H, W):
        self.amp_x = rng.uniform(*self.amp_x_range)
        self.amp_y = rng.uniform(*self.amp_y_range)
        self.freq_x = rng.uniform(*self.freq_range)
        self.freq_y = rng.uniform(*self.freq_range)

    def apply_points(self, pts, H, W):
        pts = pts.copy()
        x = pts[:, 0]
        y = pts[:, 1]

        pts[:, 0] += self.amp_x * np.sin(2*np.pi*y/H*self.freq_y)
        pts[:, 1] += self.amp_y * np.sin(2*np.pi*x/W*self.freq_x)

        return pts
