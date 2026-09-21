# C++ Port of Tennis-Player-Detection

A faithful C++17 rewrite of the Python pipeline in `../src/`, kept side by
side with the original. Every feature is mirrored module-for-module:

| Python (original) | C++ (this port) |
|---|---|
| `src/config.py` | `include/config.hpp`, `src/config.cpp` |
| `src/capture.py` (`ThreadedCapture`) | `include/capture.hpp`, `src/capture.cpp` |
| `src/tflite_detector.py` (`TFLiteDetector`, letterbox, YOLO + post-processed heads) | `include/tflite_detector.hpp`, `src/tflite_detector.cpp` |
| `src/detector.py` (`YOLODetector` ONNX) | `include/detector.hpp`, `src/detector.cpp` |
| `src/geometry.py` (pan angle) | `include/geometry.hpp`, `src/geometry.cpp` |
| `src/distance_formula.py` | `include/distance_formula.hpp`, `src/distance_formula.cpp` |
| `src/distance_estimator.py` (full height + YuNet head fallback) | `include/distance_estimator.hpp`, `src/distance_estimator.cpp` |
| `src/main.py` (CLI, frame-skip, FPS, GUI overlays) | `src/main.cpp` |
| `tools/benchmark.py` | `tools/benchmark.cpp` |
| `tools/export_models.py` | `tools/export_models.cpp` |

The engine auto-selection from `main.py::create_detector` (TFLite preferred,
ONNX fallback) lives in `include/engine_factory.hpp` / `src/engine_factory.cpp`.

---

## 1. Dependencies

| Library | Needed for | PC | Raspberry Pi |
|---|---|---|---|
| **OpenCV ≥ 4.5.4** (`core imgproc imgcodecs videoio highgui objdetect`) | capture, GUI, YuNet (`FaceDetectorYN` is in `objdetect`) | `sudo apt install libopencv-dev` | `sudo apt install libopencv-dev` |
| **ONNX Runtime ≥ 1.19** (C++ API) | the default `yolov8n_320.onnx` model (exported with ONNX opset 20) | prebuilt x64 `.tgz` | prebuilt aarch64 `.tgz` |
| **TensorFlow Lite** (optional) | fast int8 models (SSD-MobileNet, EfficientDet) | build from source (optional) | build from source (recommended) |
| **libcurl** (optional) | `export_models` only | `libcurl4-openssl-dev` | `libcurl4-openssl-dev` |

> The YuNet face model runs inside OpenCV's own DNN module, so it does **not**
> need the ONNX Runtime dependency.

> The YOLO models in this repo were exported with ONNX opset 20, so ONNX
> Runtime **1.19 or newer** is required (older versions refuse opset 20
> models with a loader error).

### ONNX Runtime (PC, x86-64)

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar xzf onnxruntime-linux-x64-1.20.1.tgz   # -> ~/onnxruntime-linux-x64-1.20.1
```

### ONNX Runtime (Pi Zero 2 W, ARM64)

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-aarch64-1.20.1.tgz
tar xzf onnxruntime-linux-aarch64-1.20.1.tgz  # -> ~/onnxruntime-linux-aarch64-1.20.1
```

> Check for a newer release; the arm64 build is what runs on the Pi's 64-bit OS.

### TensorFlow Lite (optional, for the fast int8 models)

Build the standalone TFLite library (the same one the Python `tflite-runtime`
uses under the hood). On a PC or on the Pi:

```bash
git clone --depth 1 https://github.com/tensorflow/tensorflow.git
cd tensorflow/lite/tools/make
./download_dependencies.sh
./build_rpi_lib.sh          # produces gen/rpi_armv7/lib/libtensorflow-lite.a
```

You then point CMake at **both** the source root (for headers) and the build
directory (for the library):

```
-DTENSORFLOW_LITE_SRC_ROOT=~/tensorflow
-DTENSORFLOW_LITE_LIB_DIR=~/tensorflow/lite/tools/make/gen/rpi_armv7/lib
```

---

## 2. Build

```bash
cd cpp
cmake -S . -B build \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DTENSORFLOW_LITE_SRC_ROOT=~/tensorflow \        # optional
  -DTENSORFLOW_LITE_LIB_DIR=.../gen/rpi_armv7/lib  # optional
cmake --build build -j
```

Options:

| CMake option | Default | Meaning |
|---|---|---|
| `TP_ENABLE_ONNX` | `ON` | Build the ONNX YOLO backend (requires ONNXRuntime). |
| `TP_ENABLE_TFLITE` | `ON` | Build the TFLite backend (skips gracefully if TFLite is not found). |

The three binaries land in `cpp/build/`:

| Binary | Mirrors | Purpose |
|---|---|---|
| `tennis_detector` | `python3 src/main.py` | Real-time detection loop |
| `benchmark` | `tools/benchmark.py` | Per-model FPS benchmark |
| `export_models` | `tools/export_models.py` | Downloads the int8 TFLite models |

---

## 3. Testing on a PC (no camera required)

### 3.1 Run the pipeline on a video file

```bash
./build/tennis_detector --file ../video/VID_*.mp4
# or, for an explicit model:
./build/tennis_detector --file ../video/VID_*.mp4 --model yolov8n.onnx
```

A window opens with the box + player center (red) + image center (blue) + pan
angle overlay. The terminal prints:

```
Current rotation: -20.13 degrees | Distance: 15.98 m | FPS: 25.4
```

`q` quits; add `--no-gui` to run headless (use `Ctrl+C`).

### 3.2 Test every model / engine

All models already live in `models/` at the repo root:

```bash
./build/tennis_detector --file ../video/VID_*.mp4 --model yolov8n_320.onnx                 # ONNX @ 320 (default)
./build/tennis_detector --file ../video/VID_*.mp4 --model yolov8n.onnx                     # ONNX @ 640 (accurate, slow)
./build/tennis_detector --file ../video/VID_*.mp4 --model ssd_mobilenet_v2_coco_int8_300.tflite      # TFLite @ 300
./build/tennis_detector --file ../video/VID_*.mp4 --model efficientdet_lite0_coco_int8_320.tflite    # TFLite @ 320
```

Pick the fastest with the benchmark (synthetic frame, camera-independent):

```bash
./build/benchmark --frames 100 --threads 4
```

### 3.3 Tuning knobs

| Flag | Meaning |
|---|---|
| `--camera N` | camera index (default from `include/config.hpp`, `0`) |
| `--file PATH` | run from a video file |
| `--model NAME` | model file inside `models/` |
| `--threads N` | inference threads (Pi Zero 2 W: 4) |
| `--skip N` | run inference every `N+1` frames (holds last position) |
| `--no-gui` | headless: no window/overlays |

All other settings (capture size, FPS, MJPEG, FOV, player height, confidence,
frame-skip) are in `include/config.hpp` — defaults match `src/config.py`.

### 3.4 Use the modules in your own tests

The code splits into two static libraries:

- `tp_core` — capture, geometry, distance (no ML dependencies).
- `tp_detectors` — ONNX/TFLite detectors + engine factory.

Example `test_pc.cpp` that exercises the math modules directly:

```cpp
#include <cstdio>
#include "config.hpp"
#include "detector.hpp"
#include "distance_estimator.hpp"
#include "geometry.hpp"

int main() {
    const tp::Config cfg;
    tp::CameraGeometry geometry(cfg.horizontal_fov);
    tp::DistanceEstimator estimator(cfg.horizontal_fov, cfg.player_height);

    const int W = 640, H = 480;

    // Pan angle from a synthetic detection box.
    tp::Detection d;
    d.x1 = 120; d.y1 = 40; d.x2 = 520; d.y2 = 440;
    cv::Point center = geometry.getPlayerCenter(d);
    printf("pan angle: %.2f deg\n",
           geometry.calculateHorizontalAngle(center.x, W / 2, W));

    // Distance from full body height (position-invariant).
    printf("vertical FOV: %.2f deg\n", estimator.calculateVerticalFov(H, W));
    if (auto full = estimator.estimateFromFullHeight(d, H, W))
        printf("full-height distance: %.2f m\n", *full);

    // Truncated box -> falls back to head measurement (needs a real frame).
    d.y1 = 0;  // top cut off
    printf("truncated: %s\n", estimator.isTruncated(d, H) ? "yes" : "no");
    return 0;
}
```

Build it against the libraries (or add it to CMake as its own executable):

```bash
g++ -std=c++17 -I cpp/include test_pc.cpp \
    cpp/build/libtp_core.a cpp/build/libtp_detectors.a \
    $(pkg-config --cflags --libs opencv4) -lpthread -ldl -o test_pc
```

> The detectors themselves implement `Detector::detectPerson(frame, threshold)`
> and return a `tp::Detection`, so you can drive them from a unit test the
> same way `main.cpp` does.

---

## 4. Raspberry Pi (deployment target: Pi Zero 2 W, 64-bit OS)

### 4.1 Prerequisites

```bash
sudo apt update
sudo apt install -y build-essential cmake git wget tar libopencv-dev libcurl4-openssl-dev
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Attach a heatsink/fan (the board throttles at 85 °C) and use a powered USB hub
if the camera draws more than the port provides.

### 4.2 Build the models

Run `export_models` (or the Python `tools/export_models.py`) on a PC/Mac and
copy the `.tflite` files into `models/` on the Pi:

```bash
# on the PC:
./cpp/build/export_models
scp models/*.tflite pi@raspberrypi:~/Tennis-Player-Detection/models/
```

### 4.3 Build (natively on the Pi, ~10-20 min)

With ONNX Runtime alone (default, no TFLite needed for `yolov8n_320.onnx`):

```bash
cd cpp
cmake -S . -B build -DONNXRUNTIME_ROOT=~/onnxruntime-linux-aarch64-1.16.3
cmake --build build -j4
```

With TFLite for the fast int8 models (recommended):

```bash
cmake -S . -B build \
  -DONNXRUNTIME_ROOT=~/onnxruntime-linux-aarch64-1.16.3 \
  -DTENSORFLOW_LITE_SRC_ROOT=~/tensorflow \
  -DTENSORFLOW_LITE_LIB_DIR=~/tensorflow/lite/tools/make/gen/rpi_armv7/lib
cmake --build build -j4
```

> Cross-compiling from a PC is possible but fiddly; building on the Pi keeps
> OpenCV/ONNX/ABI versions consistent. Compile on the Pi once, then rerun
> `cmake --build build -j4` after edits.

### 4.4 Pick the fastest model

```bash
./build/benchmark --frames 100 --threads 4
```

Set the winner in `include/config.hpp` (`model_file`) or pass `--model`:

```bash
./build/tennis_detector --model ssd_mobilenet_v2_coco_int8_300.tflite
```

### 4.5 Run headless (no display)

```bash
./build/tennis_detector --no-gui --threads 4
# or from a file:
./build/tennis_detector --no-gui --file ../video/VID_*.mp4
# Ctrl+C to quit
```

`--skip 1` halves the ML cost if you need more headroom (the angle output
stays smooth because capture runs on its own thread).

### 4.6 Expected performance (same ballpark as the Python version)

| Engine | Input | Expected |
|---|---|---|
| `yolov8n_320.onnx` FP32 (ONNX) | 320 | ~3-6 FPS |
| `ssd_mobilenet_v2_coco_int8_300.tflite` | 300 | ~3-6 FPS |
| `efficientdet_lite0_coco_int8_320.tflite` | 320 | ~2-4 FPS |
| `yolov8n.onnx` FP32 (ONNX) | 640 | ~1-2 FPS |

> TFLite models run through TFLite's own op kernels. For the fastest int8
> speed on the Pi, prefer a TFLite build with the XNNPACK delegate enabled
> (this mirrors what the Python `tflite-runtime` uses automatically).

---

## 5. Module reference (public API)

| Header | Provides | Key members |
|---|---|---|
| `config.hpp` | `tp::Config` | all tunables; `projectDirectory()`, `modelPath()`, `videoPath()` |
| `capture.hpp` | `tp::ThreadedCapture` | `start()`, `stop()`, `read() -> cv::Mat`, `is_file()`, `finished()` |
| `detector.hpp` | `tp::Detection`, `tp::Detector`, `tp::YOLODetector` | `detectPerson(frame, threshold)`, `valid()` |
| `tflite_detector.hpp` | `tp::TFLiteDetector` | `detectPerson(frame, threshold)`, `inferenceSize()` |
| `engine_factory.hpp` | `tp::createDetector(...)` | engine auto-selection (TFLite -> ONNX) |
| `geometry.hpp` | `tp::CameraGeometry` | `getPlayerCenter()`, `getImageCenter()`, `calculateHorizontalAngle()` |
| `distance_formula.hpp` | free functions | `heightRatio()`, `distanceFromRatio()` |
| `distance_estimator.hpp` | `tp::DistanceEstimator` | `estimate()`, `estimateFromFullHeight()`, `estimateFromHead()`, `detectHeadHeight()`, `isTruncated()` |

## 6. Differences from the Python version (intentional)

- The ONNX backend reads the model's declared input size, so both
  `yolov8n_320.onnx` (320) and `yolov8n.onnx` (640) work without touching
  `INFERENCE_SIZE`.
- `ThreadedCapture::read()` returns a copy of the latest frame so the capture
  thread can never mutate a frame mid-processing.
- Backends are optional at build time; with neither enabled the factory
  throws a clear error instead of failing silently.
- Numeric parity of `geometry` / `distance_formula` / FOV math against the
  Python originals was verified with identical inputs (bit-identical
  distances).
