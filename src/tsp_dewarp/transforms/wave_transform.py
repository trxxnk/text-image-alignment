import numpy as np

from . import BaseTransform


class RandomWaveTransform(BaseTransform):
    def __init__(self,
                 amp_x_range=(0.01, 0.05),  # доля от ширины
                 amp_y_range=(0.01, 0.05),  # доля от высоты
                 freq_range=(0.5, 2.0)):
        super().__init__()

        self.amp_x_range = amp_x_range
        self.amp_y_range = amp_y_range
        self.freq_range = freq_range

        self.strength = 1.0

    def set_strength(self, strength: float):
        self.strength = strength

    def sample(self, rng, H, W):
        # --- нормализованные амплитуды ---
        amp_x_norm = rng.uniform(*self.amp_x_range) * self.strength
        amp_y_norm = rng.uniform(*self.amp_y_range) * self.strength

        # --- перевод в пиксели ---
        self.amp_x = amp_x_norm * W
        self.amp_y = amp_y_norm * H

        # --- частоты ---
        self.freq_x = rng.uniform(*self.freq_range)
        self.freq_y = rng.uniform(*self.freq_range)

    def apply_points(self, pts, H, W):
        pts = pts.astype(np.float32, copy=True)

        x = pts[:, 0]
        y = pts[:, 1]

        pts[:, 0] += self.amp_x * np.sin(2 * np.pi * y / H * self.freq_y)
        pts[:, 1] += self.amp_y * np.sin(2 * np.pi * x / W * self.freq_x)

        return pts
