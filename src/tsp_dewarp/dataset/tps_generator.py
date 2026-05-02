import cv2
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from src.tsp_dewarp.transforms.compose import Compose


class TPSDatasetGenerator:
    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 transform_configs: list,
                 grid_size: int = 5,
                 random_seed: int = 42):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.transform_configs = transform_configs
        self.grid_size = grid_size
        self.rng = np.random.default_rng(random_seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ===== нормализация вероятностей =====
        probs = np.array([cfg["prob"] for cfg in transform_configs], dtype=np.float32)
        self.probs = probs / probs.sum()

    # ===== GRID =====
    def _build_base_grid(self, H: int, W: int) -> np.ndarray:
        xs = np.linspace(0, W - 1, self.grid_size)
        ys = np.linspace(0, H - 1, self.grid_size)
        grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
        return grid

    # ===== выбор трансформации =====
    def _sample_config(self) -> dict:
        idx = self.rng.choice(len(self.transform_configs), p=self.probs)
        return self.transform_configs[idx]

    # ===== генерация =====
    def generate(self) -> None:
        metadata = []

        for img_path in self.input_dir.glob("*"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            H, W = img.shape[:2]

            # ===== выбор конфигурации =====
            cfg = self._sample_config()
            pipeline: Compose = cfg["pipeline"]
            difficulty: str = cfg["difficulty"]

            base_grid = self._build_base_grid(H, W)

            # ===== IDENTITY =====
            if pipeline is None:
                warped = img.copy()
                delta_norm = np.zeros_like(base_grid, dtype=np.float32)
                is_identity = True

            else:
                pipeline.sample(self.rng, H, W)

                map_x, map_y = pipeline.build_remap(H, W)

                warped = cv2.remap(
                    img,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255
                )

                warped_grid = pipeline.apply_points(base_grid, H, W)

                delta = warped_grid - base_grid

                delta_norm = delta.copy()
                delta_norm[:, 0] /= (W - 1)
                delta_norm[:, 1] /= (H - 1)

                is_identity = False

            # ===== save image =====
            warped_img_name = f"{img_path.stem}_{difficulty}{img_path.suffix}"
            out_path = self.output_dir / warped_img_name

            cv2.imwrite(str(out_path), warped)

            metadata.append({
                "original": img_path.name,
                "warped": warped_img_name,
                "deltaTPS": delta_norm.tolist(),
                "difficulty": difficulty,
                "is_identity": is_identity
            })

        # ===== SAVE Metadata =====
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


    def generate_parallel(self, num_workers: int = 4):
        img_paths = list(self.input_dir.glob("*"))

        tasks = [
            (
                img_path,
                self.output_dir,
                self.transform_configs,
                self.probs,
                self.grid_size,
                42,   # base_seed
                i
            )
            for i, img_path in enumerate(img_paths)
        ]

        metadata = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(_process_single_image, tasks),
                total=len(tasks),
                desc="Generating dataset"
            ))

        # фильтр None
        metadata = [r for r in results if r is not None]

        # ===== SAVE =====
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def _process_single_image(args):
    (
        img_path,
        output_dir,
        transform_configs,
        probs,
        grid_size,
        base_seed,
        idx
    ) = args

    rng = np.random.default_rng(base_seed + idx)

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    H, W = img.shape[:2]

    # ===== sample config =====
    cfg_idx = rng.choice(len(transform_configs), p=probs)
    cfg = transform_configs[cfg_idx]

    pipeline = cfg["pipeline"]
    difficulty = cfg["difficulty"]

    # ===== base grid =====
    xs = np.linspace(0, W - 1, grid_size)
    ys = np.linspace(0, H - 1, grid_size)
    base_grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)

    # ===== apply pipeline =====
    pipeline.sample(rng, H, W)

    map_x, map_y = pipeline.build_remap(H, W)

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

    warped_grid = pipeline.apply_points(base_grid, H, W)

    delta = warped_grid - base_grid

    delta_norm = delta.copy()
    delta_norm[:, 0] /= (W - 1)
    delta_norm[:, 1] /= (H - 1)

    is_identity = True if difficulty == "identity" else False

    # ===== save =====
    warped_img_name = f"{img_path.stem}_{difficulty}{img_path.suffix}"
    out_path = output_dir / warped_img_name

    cv2.imwrite(str(out_path), warped)

    return {
        "original": img_path.name,
        "warped": warped_img_name,
        "deltaTPS": delta_norm.tolist(),
        "difficulty": difficulty,
        "is_identity": is_identity
    }
