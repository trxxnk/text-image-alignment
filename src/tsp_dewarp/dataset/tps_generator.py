from pathlib import Path
import json
import numpy as np
import cv2

from ..transforms import Compose


class TPSDatasetGenerator:
    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 pipeline: Compose,
                 grid_size: int = 5,
                 random_seed: int = 42):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.pipeline = pipeline
        self.grid_size = grid_size
        self.rng = np.random.default_rng(random_seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_base_grid(self, H, W):
        xs = np.linspace(0, W-1, self.grid_size)
        ys = np.linspace(0, H-1, self.grid_size)
        grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
        return grid

    def generate(self, n: int = 1):
        metadata = []

        for img_path in self.input_dir.glob("*"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            H, W = img.shape[:2]

            for i in range(n):
                # 1️⃣ Сэмплируем параметры
                self.pipeline.sample(self.rng, H, W)

                # 2️⃣ Строим remap
                map_x, map_y = self.pipeline.build_remap(H, W)

                # 3️⃣ Применяем к изображению
                warped = cv2.remap(
                    img,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255
                )

                # 4️⃣ Вычисляем deltaTPS
                base_grid = self._build_base_grid(H, W)
                warped_grid = self.pipeline.apply_points(base_grid, H, W)

                delta = base_grid - warped_grid


                # нормализуем
                delta_norm = delta.copy()
                delta_norm[:, 0] /= (W - 1)
                delta_norm[:, 1] /= (H - 1)

                # 5️⃣ Сохраняем изображение
                out_path = self.output_dir / (img_path.stem + f'_warped_{i+1}' + img_path.suffix)
                cv2.imwrite(str(out_path), warped)

                metadata.append({
                    "original": str(img_path),
                    "warped": str(out_path),
                    "deltaTPS": delta_norm.tolist()
                })

        # 6️⃣ Сохраняем JSON
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
