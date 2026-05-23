"""Тесты загрузки конфига обучения из YAML."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.tps_dewarp.training import TrainConfig, load_train_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "train_default.yaml"


class TestTrainConfig(unittest.TestCase):
    def test_load_default_yaml(self) -> None:
        self.assertTrue(DEFAULT_YAML.is_file(), f"missing {DEFAULT_YAML}")
        cfg = load_train_config(DEFAULT_YAML)
        self.assertIsInstance(cfg, TrainConfig)

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


if __name__ == "__main__":
    unittest.main()
