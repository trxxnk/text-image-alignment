"""Тесты загрузки конфига обучения из YAML."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.tps_dewarp.training import TrainConfig, load_train_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "train_default.yaml"
SMOKE_YAML = REPO_ROOT / "configs" / "train_smoke.yaml"


class TestTrainConfig(unittest.TestCase):
    def test_load_default_yaml(self) -> None:
        self.assertTrue(DEFAULT_YAML.is_file(), f"missing {DEFAULT_YAML}")
        cfg = load_train_config(DEFAULT_YAML)
        self.assertIsInstance(cfg, TrainConfig)

    def test_load_smoke_yaml(self) -> None:
        self.assertTrue(SMOKE_YAML.is_file(), f"missing {SMOKE_YAML}")
        cfg = load_train_config(SMOKE_YAML)
        self.assertEqual(cfg.epochs, 1)
        self.assertEqual(cfg.warmup_epochs, 0)

    def test_required_sections_present(self) -> None:
        cfg = load_train_config(DEFAULT_YAML)
        self.assertTrue(cfg.experiment_name)
        self.assertTrue(cfg.run_name_prefix)
        self.assertTrue(cfg.dataset_dir)
        self.assertGreater(cfg.canvas_size, 0)
        self.assertGreater(cfg.train_ratio + cfg.val_ratio, 0.0)
        self.assertLessEqual(cfg.train_ratio + cfg.val_ratio, 1.0)
        self.assertGreater(cfg.batch_size, 0)
        self.assertGreater(cfg.epochs, 0)
        self.assertIn(cfg.metric_for_best, ("val_loss", "val_l2_px"))
        self.assertTrue(cfg.checkpoint_dir)
        self.assertGreater(cfg.tanh_output_scale, 0.0)
        self.assertGreaterEqual(cfg.warmup_epochs, 0)
        self.assertLess(cfg.warmup_epochs, cfg.epochs)
        self.assertIn("hard", cfg.loss_difficulty_weights)


if __name__ == "__main__":
    unittest.main()
