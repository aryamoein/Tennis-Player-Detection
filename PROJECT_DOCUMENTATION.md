# Tennis Player Detection & Tracking — Project Documentation

This document explains every module, every calculation, and the reasoning behind the design,
plus a roadmap of future advancements.

---

## 1. Project Overview

The project detects a tennis player from a camera feed, locates the player's **head**
(top of the detection box) and **body center**, and continuously reports two servo-style
angles in the terminal and on the video frame:

- **Rotation (horizontal / pan)** — the angle between the image center and the player's
  position in the horizontal axis. This is how much the camera/gimbal must pan to face
  the player.
- **Vertical angle (pitch/tilt)** — the angle between the image center and the top of
  the bounding box (the player's head). This is how much the camera must tilt.

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
[player center  +  head point (box top-center)]
        │
        v
[geometry.calculate_horizontal_angle]  →  pan angle
[vertical_angle.calculate_vertical_angle]  →  tilt angle
        │
        v
[terminal print + cv2 overlay]
```

---

## 2. Project Layout

```
.
├── .gitignore                       # excludes OS/Python/video binaries
├── README.md                        # quickstart + Pi deployment guide
├── PROJECT_DOCUMENTATION.md         # this document
├── models/
│   ├── yolov8n.onnx                          # FP32 YOLOv8n (accurate fallback)
│   ├── yolov8n.pt                            # PyTorch source weights (export reference)
│   ├── ssd_mobilenet_v2_coco_int8_300.tflite # fast int8 detector (300x300)
│   └── efficientdet_lite0_coco_int8_320.tflite # balanced int8 detector (320x320)
├── src/
│   ├── main.py               # main loop: wiring + terminal output + GUI
│   ├── config.py             # single place for every tunable constant
│   ├── capture.py            # threaded, low-latency camera capture
│   ├── tflite_detector.py    # TensorFlow Lite inference backend
│   ├── detector.py           # ONNX inference backend (fallback)
│   ├── geometry.py           # horizontal (pan) angle math
│   ├── vertical_angle.py     # vertical (tilt) angle math
│   └── tracker.py            # latest player position bookkeeping
├── tools/
│   ├── export_models.py      # PC-side: fetches int8 TFLite models
│   └── benchmark.py          # Pi-side: measures FPS per model
└── videos/                    # your own test clips (gitignored)
```

---

## 3. The Modules and the Calculations

### 3.1 `src/config.py` — one place for every knife

Centralizes the tuning constants so behaviour can be changed without touching code:

| Setting | Default | Purpose |
|---|---|---|
| `CAMERA_INDEX` | `0` | Which USB camera to open |
| `CAPTURE_WIDTH/HEIGHT/FPS` | `640 / 480 / 30` | Capture resolution and rate |
| `USE_MJPEG` | `True` | Request MJPG compression from the camera |
| `MODEL_FILE` | `ssd_mobilenet_v2_coco_int8_300.tflite` | Default detection model |
| `INFERENCE_SIZE` | `320` | only the ONNX backend; TFLite fixes its own |
| `THREADS` | `4` | Inference threads (matches Pi core count) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum person confidence |
| `FRAME_SKIP` | `0` | Run inference on every `N+1` frames |
| `HORIZONTAL_FOV` | `90` | Camera horizontal field of view in degrees |

### 3.2 `src/capture.py` — threaded capture (fights the weak CPU)

A single background thread continuously does `cv2.VideoCapture.read()` and keeps only
the newest frame in one reusable buffer (`Threading.Lock`). This means:

- Capture never blocks inference (they run in parallel).
- Stale frames are overwritten, not queued → lowest possible latency.
- MJPG + `CAP_PROP_BUFFERSIZE=1` reduce decode work and buffer lag.
- For a video file the source is throttled to its natural FPS so testing behaves
  like real playback.

**Cost avoided:** a synchronous `read()` on a 720p USB camera can idle half a weak CPU.

### 3.3 `src/tflite_detector.py` — the fast detection path

Runs an int8 TensorFlow Lite model. We use the **XNNPACK** delegate (registered
automatically) plus `num_threads = 4`; XNNPACK has heavily optimized int8 kernels for
the ARMv8 (A53) cores of the Pi Zero 2 W.

**Pre-processing – letterboxing.** The camera frame (e.g. 640x480) is resized to a
square (300 or 320) *preserving aspect ratio* and padded with a gray value:

```
scale = min(model_size / width, model_size / height)
new_w  = round(width  * scale)
new_h  = round(height * scale)
left   = (model_size - new_w) // 2
  top   = (model_size - new_h) // 2
```

The `(scale, left, top)` tuple is saved and used to map predictions back into the
original frame.

**Quantized input handling.** If the model expects uint8 (0-255) we feed the pixels
as-is; if float32, we divide by 255 and convert to `CHW`.

**YOLO-style head** (single raw tensor `[1, 84, N]`): every anchor carries
`[x,y,w,h, confidence(80 classes)]`. We iterate all candidates, read only class **0
(person)**, keep the one with the highest score above `confidence_threshold`, convert
center/width/height → `x1,y1,x2,y2`, and then un-letterbox every coordinate:

```
x_orig = (x_model - left) / scale
y_orig = (y_model - top) / scale
```

**Post-processed head** (SSD/EfficientDet, built-in NMS). The standard TFLite
`DetectionPostProcess` output order is `[boxes(1,N,4), classes(1,N), scores(1,N),
num_detections(1)]` in normalized `[ymin,xmin,ymax,xmax]` units *relative to the model
input*. We scan the top `num_detections`, select class 0, multiply by `model_size` to
obtain pixel coordinates, then apply the inverse letterbox.

### 3.4 `src/detector.py` — ONNX fallback

The original YOLOv8n ONNX path. Input resized to 640x640, BGR→RGB, normalized `/255`,
transposed to `CHW`, batched to `(1,3,640,640)`. Output `[1, 84, 8400]` is decoded the
same way as the YOLO head above. Kept as the accurate-but-slower replacement if the
TFLite stack is not installed.

### 3.5 `src/geometry.py` — horizontal (pan) angle

The blue point in the middle of the frame is the **optical center** `(w/2, h/2)`. The
**player center** `pcx, pcy` is the middle of the detection box:

```
pcx = (x1 + x2) / 2
pcy = (y1 + y2) / 2
```

Horizontal rotation, in degrees:

```
pixel_offset    = pcx - image_center_x
degree_per_pixel = horizontal_fov / image_width
angle           = pixel_offset * degree_per_pixel
```

Interpretation:

- `angle = 0` → player exactly centered (pan straight ahead).
- `angle > 0` → player and gimbal must pan **right**.
- `angle < 0` → player and gimbal must pan **left**.

This is the same math a proportional controller would need: pixels offset × °/pixel.

### 3.6 `src/vertical_angle.py` — vertical (tilt) angle

We don't know the vertical field of view, so we **derive it** from the horizontal FOV
and the frame aspect ratio (pinhole/rectilinear camera relationship):

```
φ_v = 2 · atan( tan(φ_h / 2) · (image_height / image_width) )
```

With `φ_h = 90°` and a `480/640` frame this yields ≈ 73.7° vertical. The tilt angle is
then:

```
pixel_offset     = image_center_y - head_y      # + when head is above the center
degree_per_pixel = vertical_fov / image_height
angle            = pixel_offset * degree_per_pixel
```

`head_y` is the **top** of the detection box (`y1`); `head_x = (x1 + x2)/2` is used to
draw the center-to-head line. Positive angle → camera should tilt **up**; negative →
tilt down.

### 3.7 `src/tracker.py` — bookkeeping

Records the last player-center positions (deque, bounded by `history_size`). Provides
`get_current_position()`/`update(x,y)` for state without a per-frame ML cost. (The
former stop/final-rotation logic was removed in a later cleanup step; see §6.)

### 3.8 `src/main.py` — the loop

1. Load engine automatically: tries `MODEL_FILE` (`TFLite`) → falls back to ONNX
   (ImportError/FileNotFoundError).
2. Build the capture source (camera or `--file`).
3. **Frame-skip / hold**: inference runs every `FRAME_SKIP + 1` frames; on the frames
   in between, the last known person box is reused and the two angles recomputed for
   free. This decouples the *angle output rate* from the *inference rate* — the gimbal
   gets smooth updates even if inference is the bottleneck.
4. Terminal output each processed frame:
   `Current rotation: … | Vertical angle: …`.
5. Overlay the box, player center (red), image center (blue), head (green),
   center→head line, and both angle readouts.
6. `--camera`, `--file`, `--model`, `--threads`, `--skip` are all CLI overridable.

### 3.9 `tools/export_models.py` — get the right models (PC side)

No ML toolchain installed: it just downloads two ready-made **int8** detection models
(SSD-MobileNet-V2 @300 and EfficientDet-Lite0 @320) from the Google Coral model zoo.
Includes a TLS fallback for macOS machines with a missing CA store (`certifi`).

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
- **Small input (300-320)**: 4x less compute than 640 at a modest accuracy cost.
- **Letterboxing instead of stretching**: no aspect-ratio distortion → the box
  stays on the player; coordinates map back exactly with `scale` and offsets.

### Expected throughput (estimates from published Pi benchmarks)

| Platform / engine | Numbers |
|---|---|
| Pi Zero 2 W, SSDLite-MobileNet-V2 int8, 300p | ~3-6 FPS |
| Pi Zero 2 W, EfficientDet-Lite0 int8, 320p | ~2-4 FPS |
| Pi Zero 2 W, YOLOv8n ONNX FP32 @ 640 | ~1-2 FPS |
| Pi 5 (reference), YOLOv8n int8 @ 640 | ~2-3 FPS |

---

## 5. Get the numbers running

Capture: 640x480 MJPG → inference at 300/320 → two angles out at camera rate.
All details (camera, FOV, threshold) live in `src/config.py`.

```
FROM terminal:
  python3 src/main.py                       # USB camera 0
  python3 src/main.py --file videos/your.mp4 # offline test
  python3 src/main.py --model efficientdet_lite0_coco_int8_320.tflite
```

For servo use, feed `current_angle` (pan) and `current_vertical_angle` (tilt)
into a PWM controller. Smoothing is a future enhancement (§6).

---

## 6. Future advancements

### Short term (low effort, high value)

1. **Angle smoothing for the gimbal.** Add an exponential moving average (EMA) or
   a tiny Kalman filter on both angles before printing/sending them. Stops servo
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

11. **Device acceleration options.** Colost TPU (USB), or the Hailo-8 AI HAT → 60+FPS
    int8 on a Pi 5 class host. On the Pi Zero 2 W, NCNN int8 is worth benchmarking
    against TFLite/XNNPACK.
12. **Local NVRAM config.** Read all tunables from a `config.json`/`config.yaml` on
    boot, no re-edit, hot-reload on change.
13. **Data & logging.** Log time-series (angles, FPS, temperature) to CSV or SQLite
    for offline tuning; auto-detect thermal throttling.
14. **Streaming.** Push the annotated frame over RTSP/WebRTC and the angles over
    MQTT/serial so the Pi runs headless.
15. **Auto calibration.** From a few frames of a known-size object (or the tennis
    court lines), compute the real FOV automatically instead of assuming 90°.
16. **CI + tests.** Unit tests for the geometry/letterbox math, a benchmark gate
    (fail the build if inference exceeds X ms), and a GitHub Actions job that
    regenerates the models.

---

## 7. Repo housekeeping notes

- `.gitignore` excludes `.DS_Store`, `__pycache__`, and video binaries — be intentional.
- The two int8 `.tflite` models currently come from the Google Coral model zoo
  (Apache 2.0); YOLOv8 is AGPL-3.0 (be mindful if shipping commercially).
- `yolov8n.pt` is the PyTorch source, kept as the basis for re-exporting ONNX models
  at other resolutions.