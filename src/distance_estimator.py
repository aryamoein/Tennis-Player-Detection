import math

import cv2

from distance_formula import (
    distance_from_ratio,
    height_ratio
)


class DistanceEstimator:
    """
    Estimates the player-camera distance.

    Approaches:
      1. Full body height  - the whole detection box height
                             (preferred; position-invariant).
      2. Upper body (head) - the head detected inside the
                             top of the detection box (fallback).
    """

    # ------------------------------------------------------------
    # Body proportion constants (fractions of body height)
    # ------------------------------------------------------------

    # Average head-to-body ratio (~7.5 heads tall).
    HEAD_RATIO = 1.0 / 7.5


    def __init__(self, horizontal_fov, player_height=1.75, face_model=None):
        """
        Args:
            horizontal_fov:
                Horizontal field of view of the camera in degrees.

            player_height:
                Real height of the tracked player in meters.

            face_model:
                Path to a YuNet face detection ONNX model.
                When not given, the default bundled model
                (models/yunet_face_detection.onnx) is used.
        """

        self.horizontal_fov = horizontal_fov

        self.player_height = player_height

        self._face_detector = None

        self._face_model = face_model


    def _load_face_detector(self):
        """
        Lazily build the YuNet face detector.
        """

        if self._face_detector is not None:

            return self._face_detector

        if self._face_model is None:

            # Fall back to the bundled model next to the project.

            from pathlib import Path

            default = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "yunet_face_detection.onnx"
            )

            self._face_model = str(default)

        if not cv2.os.path.exists(self._face_model):

            return None

        # Input size is updated per frame in detect_head_height,
        # so any placeholder is fine here.

        self._face_detector = cv2.FaceDetectorYN.create(
            self._face_model,
            "",
            (320, 320)
        )

        return self._face_detector


    def calculate_vertical_fov(self, image_height, image_width):
        """
        Derive the vertical field of view from the
        horizontal field of view and the frame dimensions.
        """

        horizontal_rad = math.radians(
            self.horizontal_fov / 2
        )

        vertical_rad = math.atan(
            math.tan(horizontal_rad) *
            (image_height / image_width)
        )

        return 2 * math.degrees(vertical_rad)


    def is_truncated(
        self,
        detection,
        image_height,
        tolerance=2
    ):
        """
        Check whether the detection box is cut off by the
        top or bottom edge of the frame.
        """

        y1 = detection["y1"]

        y2 = detection["y2"]

        return (
            y1 <= tolerance
            or y2 >= image_height - 1 - tolerance
        )


    # ------------------------------------------------------------
    # Approach 1: full body height
    # ------------------------------------------------------------

    def estimate_from_full_height(
        self,
        detection,
        image_height,
        image_width
    ):
        """
        Distance from the whole detection box height.

        Returns None when the box is truncated.
        """

        if self.is_truncated(
            detection,
            image_height
        ):

            return None

        box_height = (
            detection["y2"] - detection["y1"]
        )

        if box_height <= 0:

            return None

        ratio = height_ratio(
            box_height,
            image_height
        )

        vertical_fov = (
            self.calculate_vertical_fov(
                image_height,
                image_width
            )
        )

        return distance_from_ratio(
            self.player_height,
            vertical_fov,
            ratio
        )


    # ------------------------------------------------------------
    # Approach 2: upper body (head) height
    # ------------------------------------------------------------

    def detect_head_height(
        self,
        frame,
        detection
    ):
        """
        Find the head inside the top half of the detection box
        using a YuNet face detector.

        Returns:
            Head height in pixels, or None when no face is found.
        """

        detector = self._load_face_detector()

        if detector is None:

            return None

        x1 = max(detection["x1"], 0)

        y1 = max(detection["y1"], 0)

        x2 = detection["x2"]

        y2 = detection["y2"]

        # Look for the head in the top half of the box.

        head_bottom = y1 + (y2 - y1) // 2

        if head_bottom <= y1:

            return None

        height, width = frame.shape[:2]

        # FaceDetectorYN input size must match the frame.

        detector.setInputSize((width, height))

        status, faces = detector.detect(frame)

        if status is False or faces is None:

            return None

        best = None

        best_area = 0

        for face in faces:

            fx1, fy1, fw, fh = (
                face[:4].astype(int)
            )

            # Keep only faces inside the head region.

            if (
                fx1 < x1 or fy1 < y1
                or fx1 + fw > x2
                or fy1 + fh > head_bottom
            ):

                continue

            area = fw * fh

            if area > best_area:

                best_area = area

                best = fh

        if best is None or best <= 0:

            return None

        return int(best)


    def estimate_from_head(
        self,
        frame,
        detection,
        image_height,
        image_width
    ):
        """
        Distance from the head height (upper body).

        The head height in the image is detected with a face
        cascade; the real head height is estimated from the
        player's height using the head-to-body ratio.
        """

        head_px = self.detect_head_height(
            frame,
            detection
        )

        if head_px is None or head_px <= 0:

            return None

        real_head = (
            self.player_height * self.HEAD_RATIO
        )

        ratio = height_ratio(
            head_px,
            image_height
        )

        vertical_fov = (
            self.calculate_vertical_fov(
                image_height,
                image_width
            )
        )

        return distance_from_ratio(
            real_head,
            vertical_fov,
            ratio
        )


    # ------------------------------------------------------------
    # Compare and combine
    # ------------------------------------------------------------

    def estimate(
        self,
        frame,
        detection,
        image_height,
        image_width
    ):
        """
        Estimate the player-camera distance.

        The full-body-height measurement is used whenever the
        player is fully visible: it is the only approach that is
        mathematically invariant to the player's horizontal
        position (pinhole projection: image height depends on
        depth, not lateral offset). The head measurement is a
        fallback when the full body is out of frame.

        Returns:
            (combined_distance, estimates) where estimates is a
            dict of approach name -> distance for comparison.
            combined_distance is None when nothing is usable.
        """

        estimates = {}

        full = self.estimate_from_full_height(
            detection,
            image_height,
            image_width
        )

        if full is not None:

            estimates["full_height"] = full

            return full, estimates

        head = self.estimate_from_head(
            frame,
            detection,
            image_height,
            image_width
        )

        if head is not None:

            estimates["head"] = head

            return head, estimates

        return None, estimates
