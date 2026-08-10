# 🎾 Tennis-Player-Detection

Detects a tennis player from a camera feed, tracks the player, and reports two measurements in the terminal:

- **Rotation** 🎯 — horizontal angle between the image center and the player center (how far the gimbal must pan).
- **Distance** 📏 — estimated distance between the player and the camera, in meters.

The pipeline is optimized for running on a **Raspberry Pi Zero 2 W** 🥧 (64-bit OS) with a USB camera.

> 🧠 Full technical write-up (module-by-module calculations + roadmap): [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md).

## 📁 Project layout

```
src/
  config.py            # all tunable settings (camera, model, threads, FOV, ...)
  capture.py           # threaded, low-latency camera capture
  detector.py          # ONNX inference backend (fallback)
  tflite_detector.py   # TensorFlow Lite inference backend (fast, recommended)
  geometry.py          # horizontal (pan) angle math
  distance_estimator.py # distance from full height (preferred) or head (fallback)
  distance_formula.py  # pinhole camera distance math
  main.py              # main loop + terminal output + GUI
tools/
  export_models.py     # run on PC/Mac: build int8 .tflite models
  benchmark.py         # run on the Pi: pick the fastest model
models/                # yolov8n.onnx, yolov8n.pt, exported .tflite models, YuNet
videos/                # test.mp4 (dev testing)
```

## 💻 Running on a PC (testing, no camera required)

```bash
python3 src/main.py --file videos/test.mp4
```

## 🥧 Deployment on the Raspberry Pi Zero 2 W

### 1. Prepare the Pi 🔧

- Install **Raspberry Pi OS 64-bit** (Bookworm).
- Attach a heatsink/fan (the board throttles at 85 °C under sustained load). 🌡️
- Use a powered USB hub if your camera draws more than the USB port provides. 🔌
- Set the CPU governor to performance for best throughput:

```bash
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 2. Install dependencies 📦

```bash
sudo apt update
sudo apt install -y python3-opencv python3-pip
python3 -m venv --system-site-packages ~/tennis-venv
source ~/tennis-venv/bin/activate
pip install tflite-runtime
```

### 3. Build the models (run this on your PC/Mac, NOT on the Pi) 🛠️

```bash
python3 tools/export_models.py
```

No extra packages are needed (no ultralytics/TensorFlow). The script downloads two ready-to-use int8 TFLite detection models from the Google Coral model zoo:

- `models/ssd_mobilenet_v2_coco_int8_300.tflite` — 300×300, fastest. ⚡
- `models/efficientdet_lite0_coco_int8_320.tflite` — 320×320, accuracy/speed balance. ⚖️

Copy the `.tflite` files (and `yolov8n.onnx` if you want it) into `models/` on the Pi.

If the download fails with an SSL certificate error on macOS, run `pip install certifi` once (already handled in the script).

### 4. Pick the fastest model on the Pi ⏱️

```bash
python3 tools/benchmark.py --frames 100
```

The benchmark prints ms and FPS per model and names the fastest one. Set the winner in `src/config.py` (`MODEL_FILE`), or pass it on the command line:

```bash
python3 src/main.py --model efficientdet_lite0_coco_int8_320.tflite
python3 src/main.py --model yolov8n.onnx   # accurate ONNX fallback
```

### 5. Run with the USB camera 📷

```bash
python3 src/main.py
```

Useful overrides:

| Flag | Meaning |
|---|---|
| `--camera N` | camera index (default from config, `0`) |
| `--file PATH` | run from a video file instead of a camera |
| `--model NAME` | model file inside `models/` |
| `--threads N` | inference threads (Pi Zero 2 W: 4) |
| `--skip N` | skip N frames between inferences (holds last position) |
| `--no-gui` | headless: no window/overlays (for a Pi without a display) |

Press `q` to quit. 🚪 (In `--no-gui` mode use `Ctrl+C`.)

Example terminal output:

```
Current rotation: -20.13 degrees | Distance: 15.98 m
```

## 🚀 Expected performance on the Pi Zero 2 W

- **SSD-MobileNet-V2 int8 @ 300** (TFLite/XNNPACK): roughly 3-6 FPS.
- **EfficientDet-Lite0 int8 @ 320** (TFLite/XNNPACK): roughly 2-4 FPS.
- **yolov8n.onnx (FP32 @ 640)** as a fallback: roughly 1-2 FPS.

If 5+ FPS is required, prefer the light TFLite model. Because capture runs in a separate thread, the angle output stays smooth even when inference is the bottleneck. 🧵

## 📏 How the distance is estimated

- The **full-body height** is used whenever the player is fully in frame — it is mathematically invariant to the player's horizontal position (pinhole projection: image height depends on depth, not lateral offset).
- When the body is truncated (top/bottom of frame), it falls back to the **head height** found by a YuNet face detector.
- Both use the pinhole formula `x = L / (2 · X · tan(θ/2))` with the player's real height from config.

## 🎛️ Tuning knobs (`src/config.py`)

- `CAMERA_INDEX`, `CAPTURE_WIDTH/HEIGHT/FPS`, `USE_MJPEG`
- `MODEL_FILE`, `INFERENCE_SIZE` (only used by the ONNX backend; TFLite models fix their own size), `THREADS`
- `CONFIDENCE_THRESHOLD`, `FRAME_SKIP`
- `HORIZONTAL_FOV` (default 68.3° — Samsung Galaxy S24 Ultra main camera)
- `PLAYER_HEIGHT` (default 1.75 m)
