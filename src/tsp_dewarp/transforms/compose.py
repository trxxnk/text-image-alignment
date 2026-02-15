import numpy as np

from . import BaseTransform


class Compose:
    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def sample(self, rng, H, W):
        for t in self.transforms:
            t.sample(rng, H, W)

    def apply_points(self, pts, H, W):
        for t in self.transforms:
            pts = t.apply_points(pts, H, W)
        return pts

    def build_remap(self, H, W):
        grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        pts = self.apply_points(pts, H, W)

        map_x = pts[:, 0].reshape(H, W).astype(np.float32)
        map_y = pts[:, 1].reshape(H, W).astype(np.float32)

        return map_x, map_y
