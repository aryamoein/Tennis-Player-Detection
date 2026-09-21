"""
Camera FOV calibration (--calib).

Assumes a person with the configured real height
(Config.PLAYER_HEIGHT) stands at a known distance
(Config.CALIBRATION_DISTANCE) from the camera. From the
detected full-body height in pixels it derives the camera's
vertical field of view, then converts it to the horizontal
field of view using the frame aspect ratio.

The full-body height is used (not head/shoulder sizes) because
it is position-invariant: the projected height depends only on
depth, so the FOV is correct wherever the person stands.
"""

import math
import time

import cv2

from config import Config


def compute_horizontal_fov(
    player_height_px,
    image_height,
    image_width,
    player_height,
    distance
):
    """
    Estimate the horizontal field of view in degrees.

    Pinhole projection at depth d for an object of real height L
    occupying a ratio X = px / image_height of the frame:

        tan(theta_v / 2) = L / (2 * X * d)

    Horizontal fov from vertical fov and aspect ratio:

        tan(theta_h / 2) = tan(theta_v / 2) * (image_width / image_height)

    Args:
        player_height_px:
            Average detected height of the person in pixels.

        image_height, image_width:
            Frame dimensions in pixels.

        player_height:
            Real height of the person in meters.

        distance:
            Real distance to the person in meters.

    Returns:
        Horizontal field of view in degrees.
    """

    if player_height_px <= 0:

        raise ValueError("player_height_px must be positive")

    if image_height <= 0 or image_width <= 0:

        raise ValueError("image dimensions must be positive")

    if distance <= 0:

        raise ValueError("distance must be positive")

    ratio = player_height_px / image_height

    horizontal_fov = 2 * math.degrees(
        math.atan(
            (player_height / (2 * ratio * distance))
            * (image_width / image_height)
        )
    )

    return horizontal_fov


def run_calibration(detector, capture):
    """
    Detect the person over a set of frames, average the full-body
    box height, compute the horizontal FOV, print it, and update
    Config.HORIZONTAL_FOV so future runs use it.
    """

    capture.start()

    heights = []

    frame_dimensions = None

    frames_collected = 0

    start_time = time.time()

    last_status_print = 0.0

    print(
        f"Calibration: person of {Config.PLAYER_HEIGHT:.2f} m "
        f"standing at {Config.CALIBRATION_DISTANCE:.1f} m from the camera. "
        f"Collecting {Config.CALIBRATION_FRAMES} full-body detections."
    )

    while frames_collected < Config.CALIBRATION_FRAMES:

        now = time.time()

        elapsed = now - start_time

        if elapsed > Config.CALIBRATION_TIMEOUT:

            print(
                "Calibration timed out: no full-body detection "
                f"appeared within {Config.CALIBRATION_TIMEOUT:.0f} s. "
                "Make sure the person is fully in frame."
            )

            break

        frame = capture.read()

        if frame is None:

            if capture.is_file() and capture.finished():

                print(
                    "Video ended before enough detections "
                    f"({frames_collected}/{Config.CALIBRATION_FRAMES})."
                )

                break

            continue

        person = detector.detect_person(
            frame,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD
        )

        height, width = frame.shape[:2]

        if person is None:

            if now - last_status_print >= 1.0:

                print(
                    f"[{elapsed:5.1f}s] waiting for a person "
                    f"(0/{Config.CALIBRATION_FRAMES})"
                )

                last_status_print = now

            continue

        box_height = person["y2"] - person["y1"]

        truncated = (
            person["y1"] <= 2
            or person["y2"] >= height - 1 - 2
        )

        if truncated:

            print(
                f"[{elapsed:5.1f}s] skipped: box cut off at the edge "
                f"(body: truncated, {box_height} px)"
            )

            continue

        heights.append(box_height)

        frame_dimensions = (height, width)

        frames_collected += 1

        running_average = sum(heights) / len(heights)

        print(
            f"[{elapsed:5.1f}s] [{frames_collected:2d}/"
            f"{Config.CALIBRATION_FRAMES}] body: full, "
            f"height {box_height} px, running avg {running_average:.1f} px"
        )

    capture.stop()

    if not heights:

        print(
            "Calibration failed: no full-body detections. "
            "Place a person with Config.PLAYER_HEIGHT meters at "
            f"{Config.CALIBRATION_DISTANCE} m from the camera."
        )

        return None

    average_height = sum(heights) / len(heights)

    image_height, image_width = frame_dimensions

    horizontal_fov = compute_horizontal_fov(
        average_height,
        image_height,
        image_width,
        Config.PLAYER_HEIGHT,
        Config.CALIBRATION_DISTANCE
    )

    print(
        f"Calibration ({len(heights)} frames): "
        f"avg player height = {average_height:.1f} px"
    )

    print(
        f"Assumed distance = {Config.CALIBRATION_DISTANCE:.1f} m, "
        f"player height = {Config.PLAYER_HEIGHT:.2f} m"
    )

    print(
        f"Computed HORIZONTAL_FOV = {horizontal_fov:.2f} degrees"
    )

    _update_config_fov(horizontal_fov)

    return horizontal_fov


def _update_config_fov(value):
    """
    Rewrite Config.HORIZONTAL_FOV in src/config.py so the new
    value is used on the next run.
    """

    from pathlib import Path

    config_path = (
        Path(__file__).resolve().parent / "config.py"
    )

    text = config_path.read_text()

    import re

    new_text, count = re.subn(
        r"(HORIZONTAL_FOV\s*=\s*)[0-9.]+",
        lambda match: f"{match.group(1)}{value:.2f}",
        text,
        count=1
    )

    if count == 1:

        config_path.write_text(new_text)

        print(
            f"Updated src/config.py: HORIZONTAL_FOV = {value:.2f}"
        )

    else:

        print(
            "Could not auto-update src/config.py "
            "(HORIZONTAL_FOV not found)."
        )
