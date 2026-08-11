# Raspberry Pi Setup

Step-by-step setup for the C++ build on a Raspberry Pi (target: **Raspberry Pi Zero 2 W**, 64-bit OS, 4 cores).

## 1. Install system dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake git libcurl4-openssl-dev \
    libopencv-dev
```

> `libopencv-dev` pulls in the OpenCV modules we need (`core imgproc
> imgcodecs videoio highgui objdetect`) including the FFmpeg video backend,
> so playing the test videos "just works".

## 2. Clone the repository

```bash
git clone https://github.com/aryamoein/Tennis-Player-Detection.git
cd Tennis-Player-Detection/cpp
```

The `models/` directory is already part of the repo, so no downloads are
needed for the default ONNX model.

## 3. ONNX Runtime

The YOLO models are exported with **ONNX opset 20**, so ONNX Runtime
**1.19 or newer** is required. On the Pi (64-bit) use the aarch64 build:

```bash
cd ~
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-aarch64-1.20.1.tgz
tar xzf onnxruntime-linux-aarch64-1.20.1.tgz   # -> ~/onnxruntime-linux-aarch64-1.20.1
```

## 4. Optional: TensorFlow Lite (fast int8 models)

Only needed if you want to use the int8 TFLite models
(`ssd_mobilenet_v2_coco_int8_300.tflite`, `efficientdet_lite0_coco_int8_320.tflite`).
Build the standalone library once:

```bash
git clone --depth 1 https://github.com/tensorflow/tensorflow.git
cd tensorflow/lite/tools/make
./download_dependencies.sh
./build_rpi_lib.sh          # produces gen/rpi_armv7/lib/libtensorflow-lite.a
```

You can skip this — the ONNX backend is the default and works without it.

## 5. Configure and build

```bash
cd ~/Tennis-Player-Detection/cpp
cmake -S . -B build \
  -DONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-aarch64-1.20.1 \
  -DTENSORFLOW_LITE_SRC_ROOT=$HOME/tensorflow \        # optional
  -DTENSORFLOW_LITE_LIB_DIR=$HOME/tensorflow/lite/tools/make/gen/rpi_armv7/lib
cmake --build build -j4
```

Binaries land in `cpp/build/`:

| Binary | Purpose |
|---|---|
| `tennis_detector` | Real-time detection loop |
| `benchmark` | Per-model FPS benchmark |
| `export_models` | Downloads the int8 TFLite models |

## 6. Run

Live camera (default index 0, headless — no window needed on the Pi):

```bash
./build/tennis_detector --no-gui
```

Run on a video file:

```bash
./build/tennis_detector --file /path/to/test.mp4 --no-gui
```

Useful flags for the Pi Zero 2 W:

| Flag | Meaning |
|---|---|
| `--threads 4` | inference threads (the Zero 2 W has 4 cores) |
| `--skip 1` | infer every 2nd frame (holds last position, doubles FPS) |
| `--no-gui` | headless (recommended) |
| `--model yolov8n_320.onnx` | default; 640 (`yolov8n.onnx`) is slower but more accurate |

Find the fastest model with the camera-independent benchmark:

```bash
./build/benchmark --frames 100 --threads 4
```
