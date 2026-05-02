import argparse

from src.tsp_dewarp.dataset.tps_generator import TPSDatasetGenerator
from src.tsp_dewarp.transforms import (
    RandomWaveTransform,
    RandomPerspectiveTransform,
    IdentityTransform,
    Compose
)


def build_pipelines():
    identity_pipeline = Compose([
        IdentityTransform()
    ])

    perspective_pipeline = Compose([
        RandomPerspectiveTransform(
            distortion_scale_range=(0.05, 0.10)  # 5.0% - 10.0%
        )
    ])

    wave_pipeline = Compose([
        RandomWaveTransform(
            amp_x_range=(0.01, 0.03),   # 1.0% - 3.0%
            amp_y_range=(0.01, 0.03),
            freq_range=(0.5, 1.0)
        )
    ])

    combo_pipeline = Compose([
        RandomPerspectiveTransform(
            distortion_scale_range=(0.01, 0.04)
        ),
        RandomWaveTransform(
            amp_x_range=(0.01, 0.04),
            amp_y_range=(0.01, 0.04),
            freq_range=(0.6, 1.2)
        )
    ])

    transform_configs = [
        {"difficulty": "easy", "pipeline": perspective_pipeline, "prob": 0.4},
        {"difficulty": "medium", "pipeline": wave_pipeline, "prob": 0.2},
        {"difficulty": "hard", "pipeline": combo_pipeline, "prob": 0.2},
        {"difficulty": "identity", "pipeline": identity_pipeline, "prob": 0.2},
    ]

    return transform_configs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    transform_configs = build_pipelines()

    generator = TPSDatasetGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        transform_configs=transform_configs,
        grid_size=9
    )

    generator.generate_parallel(num_workers=args.workers)


if __name__ == "__main__":
    main()
