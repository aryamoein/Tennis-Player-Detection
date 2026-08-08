#!/usr/bin/env python3
"""
Benchmark detection models on the Raspberry Pi.

Measures pure inference FPS on a synthetic frame, which
makes model-to-model comparison fair and camera-independent.

Usage (on the Pi):
    python tools/benchmark.py --model models/yolov8n_int8_320.tflite
    python tools/benchmark.py --model models/yolov8n_int8_320.tflite models/ssd_mobilenet_v2_coco_int8_320.tflite --frames 100
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_DIRECTORY / "src")
)


def build_detector(model_path, threads):
    """Return the right detector for the given model file."""

    from tflite_detector import TFLiteDetector

    return TFLiteDetector(
        model_path,
        threads=threads
    )


def benchmark(model_path, threads, frames):
    """Time inference on a batch of frames."""

    try:

        detector = build_detector(
            model_path,
            threads
        )

    except ImportError as error:

        print(
            f"{model_path.name:45s} SKIP "
            f"(install tflite-runtime on the Pi: {error})"
        )

        return 0

    # Synthetic 480p frame (typical camera resolution).

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    # Warm up.

    detector.detect_person(
        frame,
        confidence_threshold=0.5
    )

    start = time.perf_counter()

    for _ in range(frames):

        detector.detect_person(
            frame,
            confidence_threshold=0.5
        )

    elapsed = time.perf_counter() - start

    fps = frames / elapsed

    print(
        f"{model_path.name:45s} "
        f"{1000 * elapsed / frames:8.1f} ms "
        f"{fps:6.2f} FPS"
    )

    return fps


def main():

    parser = argparse.ArgumentParser(
        description="Benchmark int8 TFLite models on the Pi."
    )

    parser.add_argument(
        "--model",
        nargs="+",
        default=[
            "models/ssd_mobilenet_v2_coco_int8_300.tflite",
            "models/efficientdet_lite0_coco_int8_320.tflite",
        ],
        help="Model file(s) to benchmark."
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Frames to time per model (default: 100)."
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Inference threads (default: 4)."
    )

    args = parser.parse_args()

    print(
        f"Benchmarking with {args.threads} thread(s), "
        f"{args.frames} frames each.\n"
    )

    best_model = None

    best_fps = 0

    for model in args.model:

        path = Path(model)

        if not path.exists():
            print(f"Skip (not found): {model}")
            continue

        fps = benchmark(
            path,
            threads=args.threads,
            frames=args.frames
        )

        if fps > best_fps:

            best_fps = fps

            best_model = model

    print("\nFastest model:")
    print(f"  {best_model} @ {best_fps:.1f} FPS")


if __name__ == "__main__":
    main()