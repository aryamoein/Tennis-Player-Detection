import argparse

import os
import sys
import time

import cv2

from config import Config
from capture import ThreadedCapture
from detector import YOLODetector
from tflite_detector import TFLiteDetector
from geometry import CameraGeometry
from distance_estimator import DistanceEstimator
from calibration import run_calibration
from serial_comm import SerialComm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tennis player angle tracking (Pi Zero 2 W optimized)."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index to use (overrides config)."
    )

    parser.add_argument(
        "--file",
        default=None,
        help="Run from a video file instead of a camera."
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model file name inside models/ (e.g. yolov8n_int8_320.tflite)."
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Inference threads (overrides config)."
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=None,
        help="Skip N frames between inferences (overrides config)."
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless: no window, no overlays (for a Pi without a display)."
    )

    parser.add_argument(
        "--calib",
        action="store_true",
        help="Calibrate HORIZONTAL_FOV: a person of Config.PLAYER_HEIGHT "
             "stands at Config.CALIBRATION_DISTANCE meters from the camera."
    )

    return parser.parse_args()


def create_detector(project_directory, model_file=None, threads=None):
    """
    Load the configured model.

    Prefers TensorFlow Lite (fast on the Pi Zero 2 W)
    and falls back to ONNX when TFLite is unavailable.

    Args:
        model_file:
            Optional file name inside models/. When not given,
            Config.MODEL_FILE is used (with an ONNX fallback).
    """

    if threads is None:
        threads = Config.THREADS


    def load_tflite(path):
        """Try a TFLite model, falling back to ONNX on error."""

        try:

            detector = TFLiteDetector(
                path,
                threads=threads
            )

            print(f"Using TFLite engine: {path.name}")

            return detector

        except (ImportError, FileNotFoundError) as error:

            print(
                f"TFLite unavailable ({error}); "
                "falling back to ONNX."
            )

        return None



    if model_file is not None:

        # Explicit model requested.
        path = project_directory / "models" / model_file

        if path.suffix.lower() == ".tflite":

            detector = load_tflite(path)

            if detector is not None:
                return detector

        print(f"Using ONNX engine: {path.name}")

        return YOLODetector(
            path,
            inference_size=Config.INFERENCE_SIZE
        )



    # Default model from config (TFLite preferred).

    config_path = Config.model_path(project_directory)

    if config_path.suffix.lower() == ".tflite":

        detector = load_tflite(config_path)

        if detector is not None:
            return detector

    onnx_path = (
        project_directory
        / "models"
        / "yolov8n_320.onnx"
    )

    print(f"Using ONNX engine: {onnx_path.name}")

    return YOLODetector(
        onnx_path,
        inference_size=Config.INFERENCE_SIZE
    )


def has_display():
    """
    True when a graphical display is available for OpenCV
    windows. On Linux, headless Pis run without DISPLAY or
    WAYLAND_DISPLAY, so we fall back to terminal-only output.
    """

    if sys.platform.startswith("linux"):

        return bool(
            os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
        )

    return True


def get_capture(project_directory, camera_index, file_path):
    """
    Build the capture source.

    file_path wins, then an explicit camera index,
    then the camera index from config.
    """

    if file_path:

        return ThreadedCapture(file_path)

    return ThreadedCapture(
        camera_index if camera_index is not None
        else Config.CAMERA_INDEX,
        width=Config.CAPTURE_WIDTH,
        height=Config.CAPTURE_HEIGHT,
        fps=Config.CAPTURE_FPS,
        use_mjpeg=Config.USE_MJPEG
    )


def main(args):

    project_directory = (
        Config.project_directory()
    )

    detector = create_detector(
        project_directory,
        model_file=args.model,
        threads=args.threads
    )

    capture = get_capture(
        project_directory,
        camera_index=args.camera,
        file_path=args.file
    )

    if args.calib:

        run_calibration(detector, capture)

        return

    geometry = CameraGeometry(
        horizontal_fov=Config.HORIZONTAL_FOV
    )

    distance_estimator = DistanceEstimator(
        horizontal_fov=Config.HORIZONTAL_FOV,
        player_height=Config.PLAYER_HEIGHT,
        face_model=(
            project_directory
            / "models"
            / "yunet_face_detection.onnx"
        )
    )

    serial_comm = SerialComm()

    capture.start()

    show_gui = not args.no_gui and has_display()

    if not show_gui:

        print("No display detected; running headless (terminal output only).")

    last_person = None

    frame_index = 0

    fps = 0.0

    fps_start_time = time.time()

    fps_frame_count = 0

    inference_every = (
        (args.skip if args.skip is not None
         else Config.FRAME_SKIP) + 1
    )

    last_no_player_log_time = 0.0

    no_player_log_interval = 0.5



    while True:

        frame = capture.read()

        if frame is None:

            if capture.is_file() and capture.finished():

                print("Video ended")
                break

            # Camera/file not ready yet.
            if show_gui and cv2.waitKey(1) == ord("q"):
                break

            continue



        # Run inference only every (FRAME_SKIP + 1) frames.
        # On skipped frames we reuse the last known person box.

        if frame_index % inference_every == 0:

            person = detector.detect_person(
                frame,
                confidence_threshold=Config.CONFIDENCE_THRESHOLD
            )

            last_person = person

        frame_index += 1

        fps_frame_count += 1

        now = time.time()

        if now - fps_start_time >= 1.0:

            fps = fps_frame_count / (now - fps_start_time)

            fps_frame_count = 0

            fps_start_time = now



        if last_person is None:

            if now - last_no_player_log_time >= no_player_log_interval:

                print(
                    f"No player detected | "
                    f"FPS: {fps:.1f}"
                )

                last_no_player_log_time = now

            if show_gui:

                cv2.imshow(
                    "Tennis Player Detection",
                    frame
                )

                if cv2.waitKey(1) == ord("q"):
                    break

            continue



        # Bounding box

        x1 = last_person["x1"]
        y1 = last_person["y1"]

        x2 = last_person["x2"]
        y2 = last_person["y2"]


        if show_gui:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )



        # Player center

        player_x, player_y = (
            geometry.get_player_center(
                last_person
            )
        )



        # Image center

        image_x, image_y = (
            geometry.get_image_center(
                frame
            )
        )



        # Calculate current angle

        current_angle = (
            geometry.calculate_horizontal_angle(
                player_x,
                image_x,
                frame.shape[1]
            )
        )



        # Calculate distance between player and camera

        combined_distance, estimates = (
            distance_estimator.estimate(
                frame,
                last_person,
                frame.shape[0],
                frame.shape[1]
            )
        )



        # -------------------------------
        # PRINT REAL TIME ANGLE
        # -------------------------------

        if combined_distance is None:

            print(
                f"Current rotation: {current_angle:.2f} degrees | "
                f"Distance: N/A (player out of frame) | "
                f"Angle: N/A | "
                f"Body: out of frame | "
                f"FPS: {fps:.1f}"
            )

            serial_comm.send(0, 0, 0)

        else:

            if "full_height" in estimates:

                body_flag = "full"

            else:

                body_flag = "truncated"

            angle = distance_estimator.distance_to_angle(
                combined_distance
            )

            print(
                f"Current rotation: {current_angle:.2f} degrees | "
                f"Distance: {combined_distance:.2f} m | "
                f"Angle: {angle:.2f} degrees | "
                f"Body: {body_flag} | "
                f"FPS: {fps:.1f}"
            )

            serial_comm.send(
                round(current_angle, 2),
                round(angle, 2),
                Config.SPEED
            )



        # Draw player center

        if show_gui:

            cv2.circle(
                frame,
                (player_x, player_y),
                5,
                (0, 0, 255),
                -1
            )


        # Draw image center

        if show_gui:

            cv2.circle(
                frame,
                (image_x, image_y),
                5,
                (255, 0, 0),
                -1
            )



        # Display current angle

        if show_gui:

            cv2.putText(
                frame,
                f"Angle: {current_angle:.2f} deg",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )



        if show_gui:

            cv2.imshow(
                "Tennis Player Detection",
                frame
            )

            if cv2.waitKey(1) == ord("q"):
                break



    capture.stop()

    serial_comm.close()

    if show_gui:
        cv2.destroyAllWindows()



if __name__ == "__main__":
    args = parse_args()
    main(args)