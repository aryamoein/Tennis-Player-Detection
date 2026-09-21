# Tennis Player Detection & Tracking — Project Documentation

This document explains every module, every calculation, and the reasoning behind the design,
plus a roadmap of future advancements.

---

## 1. Project Overview

The project detects a tennis player from a camera feed, locates the player's **bounding box**,
and continuously reports two measurements in the terminal (and optionally on the video frame):

- **Rotation (horizontal / pan)** — the angle between the image center and the player's
  horizontal position. This is how much the camera/gimbal must pan to face the player.
- **Distance** — the estimated distance between the player and the camera, in meters,
  computed from the player's apparent height (full body, or head as a fallback).

The primary deployment target is a **Raspberry Pi Zero 2 W** (quad-core Cortex-A53
@1 GHz, 512 MB RAM, 64-bit OS) with a USB camera, so the pipeline is heavily optimized
for very weak hardware.

```
[USB camera 640x480 MJPG]
        │  threaded capture (ThreadedCapture, drops stale frames)
        v
[TFLite int8 detector @ 300/320 px, XNNPACK, 4 threads]  ── or ──> [ONNX yolov8n fallback]
        │  (single best "person" box)
        v
[player box  →  player center]
        │
        v
[geometry.CameraGeometry]  →  pan angle (degrees)
[distance_estimator.estimate]  →  distance (full height  →  head fallback)
        │
        v
[terminal print (angle + distance + FPS) + cv2 overlay]
```

---

## 2. Project Layout

```
.
├── .gitignore                       # excludes OS/Python/video binaries
├── README.md                        # quickstart + Pi deployment guide
├── PROJECT_DOCUMENTATION.md         # this document
├── models/
│   ├── yolov8n_320.onnx                     # FP32 YOLOv8n @ 320 (default, no TFLite needed)
│   ├── yolov8n.onnx                         # FP32 YOLOv8n @ 640 (accurate fallback)
│   ├── yolov8n.pt                           # PyTorch source weights (export reference)
│   ├── yunet_face_detection.onnx            # YuNet face detector for the head fallback
│   ├── ssd_mobilenet_v2_coco_int8_300.tflite # fast int8 detector (300x300)
│   └── efficientdet_lite0_coco_int8_320.tflite # balanced int8 detector (320x320)
├── src/
│   ├── main.py               # main loop: wiring + terminal output + GUI
│   ├── config.py             # single place for every tunable constant
│   ├── capture.py            # threaded, low-latency camera capture
│   ├── tflite_detector.py    # TensorFlow Lite inference backend
│   ├── detector.py           # ONNX inference backend (fallback)
│   ├── geometry.py           # horizontal (pan) angle math
│   ├── distance_formula.py   # pinhole camera distance math
│   └── distance_estimator.py # full-height (preferred) + head (fallback) distance
├── tools/
│   ├── export_models.py      # PC-side: fetches int8 TFLite models
│   └── benchmark.py          # Pi-side: measures FPS per model
└── videos/                    # your own test clips (gitignored)
```

---

## 3. The Modules and the Calculations

### 3.1 `src/config.py` — one place for every knife

Centralizes the tuning constants so behaviour can be changed without touching code.
All settings live on a `Config` class and are addressed as `Config.X`.

| Setting | Default | Purpose |
|---|---|---|
| `CAMERA_INDEX` | `0` | Which USB camera to open |
| `VIDEO_PATH` | `videos/test.mp4` | Video file used for offline development |
| `CAPTURE_WIDTH/HEIGHT/FPS` | `640 / 480 / 30` | Capture resolution and rate |
| `USE_MJPEG` | `True` | Request MJPG compression from the camera |
| `MODEL_FILE` | `yolov8n_320.onnx` | Default detection model (ONNX @ 320, no TFLite needed) |
| `INFERENCE_SIZE` | `320` | Only the ONNX backend; TFLite fixes its own size |
| `THREADS` | `4` | Inference threads (matches Pi core count) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum person confidence |
| `FRAME_SKIP` | `0` | Run inference on every `N+1` frames |
| `HORIZONTAL_FOV` | `68.3` | Camera horizontal FOV in degrees (Galaxy S24 Ultra main) |
| `PLAYER_HEIGHT` | `1.75` | Real height of the tracked player in meters |

`Config` also exposes path helpers (`project_directory()`, `model_path()`,
`video_path()`) so modules resolve filesystem paths relative to the project root
instead of the current working directory.

### 3.2 `src/capture.py` — threaded capture (fights the weak CPU)

A single background thread continuously does `cv2.VideoCapture.read()` and keeps only
the newest frame in one reusable buffer (`threading.Lock`). This means:

- Capture never blocks inference (they run in parallel).
- Stale frames are overwritten, not queued → lowest possible latency.
- MJPG + `CAP_PROP_BUFFERSIZE=1` reduce decode work and buffer lag.
- For a video file the source is throttled to its natural FPS so testing behaves
  like real playback and the loop detects end-of-file (`finished()`).

**Cost avoided:** a synchronous `read()` on a 720p USB camera can idle half a weak CPU.

### 3.3 `src/tflite_detector.py` — the fast detection path

Runs an int8 TensorFlow Lite model. We use the **XNNPACK** delegate (registered
automatically) plus `num_threads = 4`; XNNPACK has heavily optimized int8 kernels for
the ARMv8 (A53) cores of the Pi Zero 2 W. Imports are lazy (`tflite_runtime` first,
then `tensorflow.lite`), so the rest of the app still runs where TFLite is not installed.

**Pre-processing – letterboxing.** The camera frame (e.g. 640x480) is resized to a
square (300 or 320) *preserving aspect ratio* and padded with a gray value:

```
scale = min(model_size / width, model_size / height)
new_w  = round(width  * scale)
new_h  = round(height * scale)
left   = (model_size - new_w) // 2
top    = (model_size - new_h) // 2
```

The `(scale, left, top)` tuple is saved and used to map predictions back into the
original frame.

**Quantized input handling.** If the model expects uint8 (0-255) the pixels are fed
as-is; if it expects float32, the blob is divided by `255.0`. Models are trained on
RGB, so each frame is converted from BGR first.

**YOLO-style head** (single raw tensor `[1, 84, N]`): every anchor carries
`[x,y,w,h, confidence(80 classes)]`. We iterate all candidates, read only class **0
(person)**, keep the one with the highest score above `confidence_threshold`, convert
center/width/height → `x1,y1,x2,y2`, and then un-letterbox every coordinate:

```
x_orig = (x_model - left) / scale
y_orig = (y_model - top)  / scale
```

**Post-processed head** (SSD/EfficientDet, built-in NMS). The standard TFLite
`DetectionPostProcess` output order is `[boxes(1,N,4), classes(1,N), scores(1,N),
num_detections(1)]` in normalized `[ymin,xmin,ymax,xmax]` units *relative to the model
input*. We scan the top `num_detections`, select class 0, multiply by `model_size` to
obtain pixel coordinates, then apply the inverse letterbox.

### 3.4 `src/detector.py` — ONNX fallback

The original YOLOv8n ONNX path. The frame is resized to a square
`INFERENCE_SIZE` (320 by default), BGR→RGB, normalized `/255`, transposed to `CHW`,
and batched to `(1,3,size,size)`. The output `[1, 84, N]` is decoded the same way as
the YOLO head above, and the box is scaled back by `original / inference_size` in each
axis. Kept as the accurate-but-slower replacement; it is also the only backend that
runs with zero extra dependencies beyond `onnxruntime`.

### 3.5 `src/geometry.py` — horizontal (pan) angle

`CameraGeometry` holds the horizontal FOV and provides three helpers. The blue point
in the middle of the frame is the **optical center** `(w/2, h/2)`. The **player center**
`pcx, pcy` is the middle of the detection box:

```
pcx = (x1 + x2) / 2
pcy = (y1 + y2) / 2
```

Horizontal rotation, in degrees:

```
pixel_offset     = pcx - image_center_x
degree_per_pixel = horizontal_fov / image_width
angle            = pixel_offset * degree_per_pixel
```

Interpretation:

- `angle = 0` → player exactly centered (pan straight ahead).
- `angle > 0` → player and gimbal must pan **right**.
- `angle < 0` → player and gimbal must pan **left**.

This is the same math a proportional controller would need: pixels offset × °/pixel.

### 3.6 `src/distance_formula.py` — pinhole distance math

Two pure functions shared by every distance approach:

```
height_ratio(player_height_px, image_height_px) = player_height_px / image_height_px
```

and the pinhole camera distance formula:

```
x = L / (2 · X · tan(θ / 2))
```

Where:

- `x` = distance between the player and the camera (meters)
- `L` = the player's real height (meters)
- `X` = ratio of the player's height in the image (0..1)
- `θ` = the camera's **vertical** field of view (degrees)

Both functions validate their inputs (`image_height_px` and `height_ratio_value`
positive, `vertical_fov` inside `(0, 180)`).

### 3.7 `src/distance_estimator.py` — the two distance approaches

`DistanceEstimator` produces the distance measurement. It holds the player's real
height, the horizontal FOV, and a lazily-built **YuNet** face detector
(`cv2.FaceDetectorYN`) used by the fallback path.

**Vertical FOV.** We don't know the vertical field of view, so we derive it from the
horizontal FOV and the frame aspect ratio (pinhole/rectilinear relationship):

```
φ_v = 2 · atan( tan(φ_h / 2) · (image_height / image_width) )
```

With `φ_h = 68.3°` and a `480/640` frame this yields ≈ 52.8° vertical.

**Approach 1 — full body height (preferred).** The whole detection-box height is used:

```
ratio  = box_height / image_height
distance = player_height / (2 · ratio · tan(φ_v / 2))
```

A box is considered **truncated** when its top or bottom touches the frame edges
(`y1 <= tolerance` or `y2 >= image_height - 1 - tolerance`); truncated boxes return
`None`. Full height is the only approach that is mathematically **invariant to the
player's horizontal position** — pinhole projection: image height depends on depth,
not lateral offset.

**Approach 2 — head height (fallback).** Only used when the full body is out of
frame. A YuNet face model searches the top half of the detection box for the largest
face, whose height is `head_px`. The real head height is approximated from the
player's height with the average head-to-body ratio (~7.5 heads tall):

```
real_head = player_height · (1 / 7.5)
distance  = real_head / (2 · head_ratio · tan(φ_v / 2))
```

**`estimate()`** returns `(combined_distance, estimates)`:
`estimates` holds every approach's own result keyed by name for comparison, and
`combined_distance` is the first usable measurement (full height first, then head)
or `None`. `None` means "player out of frame / nothing usable" and the main loop
prints `Distance: N/A`.

### 3.8 `src/main.py` — the loop

1. Pick the engine in `create_detector()`: an explicit `--model` (TFLite preferred,
   `ImportError`/`FileNotFoundError` falls back to ONNX), otherwise
   `Config.MODEL_FILE` (TFLite), otherwise the bundled `yolov8n_320.onnx` ONNX.
2. Build the capture source (`get_capture`): `--file` wins, then `--camera`, then
   `Config.CAMERA_INDEX`; `ThreadedCapture` starts its background thread.
3. **Frame-skip / hold**: inference runs every `FRAME_SKIP + 1` frames; on the frames
   in between, the last known person box is reused and the angle/distance recomputed
   for free. This decouples the *measurement output rate* from the *inference rate*.
4. A 1-second sliding window measures **FPS** from processed frames.
5. Terminal output each processed frame:
   `Current rotation: … degrees | Distance: … m | FPS: …` (Distance: `N/A (player
   out of frame)` when nothing is usable).
6. GUI overlay (skipped entirely with `--no-gui`): green bounding box, red player
   center, blue image center, and the current pan angle text.
7. `--camera`, `--file`, `--model`, `--threads`, `--skip`, `--no-gui` are all CLI
   overridable. `q` quits in GUI mode; `Ctrl+C` in headless mode.

### 3.9 `tools/export_models.py` — get the right models (PC side)

No ML toolchain installed: it just downloads two ready-made **int8** detection models
(SSD-MobileNet-V2 @300 and EfficientDet-Lite0 @320) from the Google Coral model zoo.
Includes a TLS fallback for machines with a missing CA store (`certifi`).

### 3.10 `tools/benchmark.py` — honest FPS on the Pi

Runs a fixed number of inferences on a synthetic 480p frame and reports:

```
inference_ms = (end - start) / frames × 1000
fps          = frames / elapsed_total
```

This makes model-vs-model comparison unbiased (no camera quirks). The fastest model is
then set in `config.py`.

---

## 4. Why int8 + TFLite + letterboxing at 320?

- **int8 quantized**: 4x fewer bytes per multiply, XNNPACK int8 kernels on A53, and
  model files ~3-6 MB. In practice this is the difference between not usable and
  usable on a Pi Zero 2 W.
- **Small input (300-320)**: 4x less compute than 640 at a modest accuracy cost. The
  default `yolov8n_320.onnx` gives the same benefit with zero extra packages.
- **Letterboxing instead of stretching**: no aspect-ratio distortion → the box
  stays on the player; coordinates map back exactly with `scale` and offsets.

### Expected throughput (estimates from published Pi benchmarks)

| Platform / engine | Numbers |
|---|---|
| Pi Zero 2 W, YOLOv8n ONNX FP32 @ 320 (default) | ~3-6 FPS |
| Pi Zero 2 W, SSDLite-MobileNet-V2 int8, 300p | ~3-6 FPS |
| Pi Zero 2 W, EfficientDet-Lite0 int8, 320p | ~2-4 FPS |
| Pi Zero 2 W, YOLOv8n ONNX FP32 @ 640 | ~1-2 FPS |
| Pi 5 (reference), YOLOv8n int8 @ 640 | ~2-3 FPS |

---

## 5. Get the numbers running

Capture: 640x480 MJPG → inference at 300/320 → angle + distance out at camera rate.
All details (camera, FOV, threshold, player height) live in `src/config.py`.

```
FROM terminal:
  python3 src/main.py                        # USB camera 0, default yolov8n_320.onnx
  python3 src/main.py --file videos/your.mp4 # offline test
  python3 src/main.py --model efficientdet_lite0_coco_int8_320.tflite
  python3 src/main.py --no-gui               # headless (no window/overlays)
```

For servo use, feed `current_angle` (pan) and `combined_distance` into a controller.
Smoothing is a future enhancement (§6).

---

## 6. Future advancements

### Short term (low effort, high value)

1. **Angle smoothing for the gimbal.** Add an exponential moving average (EMA) or
   a tiny Kalman filter on the angle before printing/sending it. Stops servo
   jitter when the head/box bounces.
2. **ByteTrack / SORT tracking.** A real association tracker would keep a stable
   identity and reject false boxes; the angles become monotonic.
3. **Optical-flow interpolation.** Between detections, track a few feature points
   (cv2.goodFeaturesToTrack + Lucas-Kanade) to *predict* the player position and the
   angle at the capture rate, making the gimbal smooth even at 2-3 FPS inference.
4. **CSI camera on the Pi.** The native CSI interface (picamera2) has far less
   decode overhead than a USB cam.
5. **Frame-skip tuning.** `FRAME_SKIP=1` halves ML cost; the hold+interpolation
   keeps the output fluid.
6. **Low-pass on the box.** Smooth the box coordinates across frames before
   geometry (removes box jitter caused by confidence changes).

### Medium term

7. **Servo/gimbal closed loop.** Map degrees to PWM duty, add PID on the error so the
   gimbal converges cleanly, and add limit/crush protection.
8. **Camera calibration.** Account for lens distortion / intrinsic matrix so pixel angles
   are accurate at the edges (currently the linear °/pixel approximation is
   best in the center).
9. **Multi-person.** Track 2-4 players and add a "current target" selector
   (closest to center / largest box / manual index).
10. **Head pose / keypoint model.** Swap the bbox for YOLOv8-pose or MediaPipe to
    locate the neck/head keypoint → a more precise and stable head reference.

### Longer term / production

11. **Device acceleration options.** Coral TPU (USB), or the Hailo-8 AI HAT → 60+FPS
    int8 on a Pi 5 class host. On the Pi Zero 2 W, NCNN int8 is worth benchmarking
    against TFLite/XNNPACK.
12. **Local NVRAM config.** Read all tunables from a `config.json`/`config.yaml` on
    boot, no re-edit, hot-reload on change.
13. **Data & logging.** Log time-series (angles, distance, FPS, temperature) to CSV or
    SQLite for offline tuning; auto-detect thermal throttling.
14. **Streaming.** Push the annotated frame over RTSP/WebRTC and the angles over
    MQTT/serial so the Pi runs headless.
15. **Auto calibration.** From a few frames of a known-size object (or the tennis
    court lines), compute the real FOV automatically instead of assuming a fixed value.
16. **CI + tests.** Unit tests for the geometry/letterbox/distance math, a benchmark
    gate (fail the build if inference exceeds X ms), and a GitHub Actions job that
    regenerates the models.

---

## 7. Repo housekeeping notes

- `.gitignore` excludes `.DS_Store`, `__pycache__`, and video binaries — be intentional.
- The two int8 `.tflite` models currently come from the Google Coral model zoo
  (Apache 2.0); YOLOv8 is AGPL-3.0 (be mindful if shipping commercially).
- `yolov8n.pt` is the PyTorch source, kept as the basis for re-exporting ONNX models
  at other resolutions (e.g. `yolov8n_320.onnx`).
- `yunet_face_detection.onnx` is OpenCV's bundled YuNet, used only by the head-height
  distance fallback when the player is out of frame.