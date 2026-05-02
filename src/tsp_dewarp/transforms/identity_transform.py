from . import BaseTransform


class IdentityTransform(BaseTransform):
    def sample(self, rng, H, W):
        pass

    def apply_points(self, pts, H, W):
        return pts
