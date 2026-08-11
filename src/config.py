from pathlib import Path


class Config:
    """
    Central place for all tunable settings.

    Optimized for Raspberry Pi Zero 2 W (64-bit OS).
    """

    # ------------------------------------------------------------
    # Video source
    # ------------------------------------------------------------
    # camera index (0, 1, ...) for a USB camera.
    # Or set CAMERA_INDEX to None and point VIDEO_PATH
    # at a file to run from a video instead.
    CAMERA_INDEX = 0

    VIDEO_PATH = Path("videos") / "test.mp4"

    CAPTURE_WIDTH = 640
    CAPTURE_HEIGHT = 480
    CAPTURE_FPS = 30
    USE_MJPEG = True

    # ------------------------------------------------------------
    # Detection model
    # ------------------------------------------------------------
    # Supported extensions:
    #   .tflite -> TFLiteDetector (fast, recommended on the Pi)
    #   .onnx   -> YOLODetector   (fallback)
    #
    # Default is the ONNX yolov8n exported at 320x320 so the
    # program runs without tflite-runtime installed.
    MODEL_FILE = Path("models") / "yolov8n_320.onnx"

    # Inference resolution. 320 is a good balance for the
    # Pi Zero 2 W. Use 224 for even more speed.
    # Only the ONNX backend; TFLite models fix their own size.
    INFERENCE_SIZE = 320

    # Number of inference threads (Pi Zero 2 W has 4 cores).
    THREADS = 4

    CONFIDENCE_THRESHOLD = 0.5

    # Run inference on every (FRAME_SKIP + 1) frames.
    # On skipped frames the last known position is held.
    FRAME_SKIP = 0

    # ------------------------------------------------------------
    # Camera geometry
    # ------------------------------------------------------------
    # Samsung Galaxy S24 Ultra main camera (rear wide, 4:3):
    #   68.3 deg horizontal FOV -> ~54.0 deg vertical FOV at 4:3.
    HORIZONTAL_FOV = 72

    # FOV calibration (--calib): the assumed distance in meters
    # between the camera and the person standing in frame.
    CALIBRATION_DISTANCE = 2.0

    # FOV calibration: how many detected frames to average.
    CALIBRATION_FRAMES = 60

    # FOV calibration: give up after this many seconds if no
    # full-body detection has appeared yet.
    CALIBRATION_TIMEOUT = 30

    # ------------------------------------------------------------
    # Player
    # ------------------------------------------------------------
    # Real height of the tracked player in meters (175 cm).
    PLAYER_HEIGHT = 1.72

    # Average head-to-body ratio (~7.5 heads tall).
    # Used by the head-height distance fallback.
    HEAD_RATIO = 1.0 / 7.5

    @classmethod
    def project_directory(cls):
        return Path(__file__).resolve().parent.parent

    @classmethod
    def model_path(cls, project_directory=None):
        if project_directory is None:
            project_directory = cls.project_directory()
        return project_directory / cls.MODEL_FILE

    @classmethod
    def video_path(cls, project_directory=None):
        if project_directory is None:
            project_directory = cls.project_directory()
        return project_directory / cls.VIDEO_PATH