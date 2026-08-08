#!/usr/bin/env python3
"""
Build detection models for the Raspberry Pi Zero 2 W.

Run this on a PC or Mac, NOT on the Pi.

No ultralytics, TensorFlow, or LiteRT is required.
This script only downloads two ready-to-use int8 TFLite
detection models from the Google Coral model zoo:

  1. ssd_mobilenet_v2_coco_int8_300.tflite   (300x300 input)
  2. efficientdet_lite0_coco_int8_320.tflite (320x320 input)

Both are used by TFLiteDetector on the Pi (via tflite-runtime,
XNNPACK accelerated) and are tuned for very weak boards.

The existing models/yolov8n.onnx stays as the accurate ONNX
fallback (run it with: python src/main.py --model yolov8n.onnx).

Usage:
    python3 tools/export_models.py
"""

from pathlib import Path
import shutil
import ssl
import urllib.request

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent

MODELS_DIRECTORY = PROJECT_DIRECTORY / "models"

CORAL_BASE = (
    "https://github.com/google-coral/test_data/raw/master/"
)

MODELS = {
    "ssd_mobilenet_v2_coco_int8_300.tflite": (
        CORAL_BASE
        + "ssd_mobilenet_v2_coco_quant_postprocess.tflite"
    ),
    "efficientdet_lite0_coco_int8_320.tflite": (
        CORAL_BASE
        + "efficientdet_lite0_320_ptq.tflite"
    ),
}


def ssl_context():
    """Best effort SSL context (fixes a missing CA store)."""

    try:

        import certifi

        return ssl.create_default_context(
            cafile=certifi.where()
        )

    except Exception:

        return ssl.create_unverified_context()


def download(url, target):
    """Download a file with a TLS fallback."""

    print(f"{target.name}")

    context = ssl_context()

    with (
        urllib.request.urlopen(url, context=context) as response
    ):

        with open(target, "wb") as handle:
            shutil.copyfileobj(response, handle)


def main():

    MODELS_DIRECTORY.mkdir(exist_ok=True)

    for name, url in MODELS.items():

        target = MODELS_DIRECTORY / name

        if target.exists():
            print(f"  {name}: already present")
            continue

        download(url, target)

        print(
            f"  -> {target.relative_to(PROJECT_DIRECTORY)}"
        )

    print(
        "Done. Benchmark the models on the Pi:\n"
        "  python3 tools/benchmark.py\n\n"
        "Pick the fastest and set it in src/config.py "
        "(MODEL_FILE), or pass --model on the command line."
    )


if __name__ == "__main__":
    main()