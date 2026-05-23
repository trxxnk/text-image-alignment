import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tps_dewarp.dataset.canvas_spatial import (
    CanvasSpatialSpec,
    DeltaTPSNormCanvasTransform,
    WarpedImageCanvasTransform,
    build_base_grid,
    uint8_hw_to_float01_chw,
)
from src.tps_dewarp.transforms.letterbox_resize import LetterboxResize
from torchvision.transforms import v2


class TestCanvasSpatial(unittest.TestCase):
    def test_identity_delta_stays_zero_letterbox(self):
        spec = CanvasSpatialSpec(32, 32, mode="letterbox", fill=0.0)
        tps_tf = DeltaTPSNormCanvasTransform(grid_size=9)
        h, w = 48, 64
        ctx = spec.build(h, w)
        delta = torch.zeros(81, 2)
        out = tps_tf(delta, h, w, ctx)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-5))

    def test_letterbox_image_matches_letterbox_resize(self):
        size = 32
        h_in, w_in = 48, 64
        raw = (np.random.rand(h_in, w_in) * 255).astype(np.uint8)
        img01 = uint8_hw_to_float01_chw(raw)
        spec = CanvasSpatialSpec(size, size, mode="letterbox", fill=0.0)
        ctx = spec.build(h_in, w_in)
        out_custom = WarpedImageCanvasTransform(fill=spec.fill)(img01, ctx)

        legacy = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                LetterboxResize(size),
            ]
        )
        out_legacy = legacy(raw)
        self.assertEqual(out_custom.shape, out_legacy.shape)
        self.assertTrue(torch.allclose(out_custom, out_legacy, atol=1e-5))

    def test_stretch_delta_smoke(self):
        out_h, out_w = 32, 48
        h_in, w_in = 16, 24
        spec = CanvasSpatialSpec(out_h, out_w, mode="stretch")
        ctx = spec.build(h_in, w_in)
        tps_tf = DeltaTPSNormCanvasTransform(grid_size=5)
        delta = torch.zeros(25, 2)
        delta[0, 0] = 0.1
        out = tps_tf(delta, h_in, w_in, ctx)
        self.assertEqual(out.shape, (25, 2))
        self.assertFalse(torch.allclose(out, torch.zeros_like(out)))

    def test_build_base_grid_order(self):
        H, W, G = 10, 12, 3
        xs = np.linspace(0, W - 1, G)
        ys = np.linspace(0, H - 1, G)
        ref = np.array([(float(x), float(y)) for y in ys for x in xs])
        got = build_base_grid(H, W, G).numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
