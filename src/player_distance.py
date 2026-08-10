import math

from distance_formula import (
    distance_from_ratio,
    height_ratio
)


class PlayerDistance:
    """
    Estimates the distance between the detected player
    and the camera from a detection box.
    """

    def __init__(self, horizontal_fov, player_height=1.75):
        """
        Args:
            horizontal_fov:
                Horizontal field of view of the camera in degrees.

            player_height:
                Real height of the tracked player in meters.
                Used when no height is passed to calculate_distance.
        """

        self.horizontal_fov = horizontal_fov

        self.player_height = player_height


    def is_truncated(
        self,
        detection,
        image_height,
        tolerance=2
    ):
        """
        Check whether the detection box is cut off by the
        top or bottom edge of the frame.

        A truncated box means the player's full height is
        not visible, so the distance estimate would be wrong
        (too large). Callers should treat the result of
        calculate_distance as unreliable in that case.

        Args:
            detection:
                Dictionary returned by detector.py.

            image_height:
                Height of the image in pixels.

            tolerance:
                How close (in pixels) a box edge may be to
                the frame edge before it counts as cut off.

        Returns:
            True when the box touches the top or bottom edge.
        """

        y1 = detection["y1"]

        y2 = detection["y2"]

        return (
            y1 <= tolerance
            or y2 >= image_height - 1 - tolerance
        )


    def calculate_vertical_fov(self, image_height, image_width):
        """
        Derive the vertical field of view from the
        horizontal field of view and the frame dimensions.

        Relationship:
            tan(vertical_fov / 2) =
                tan(horizontal_fov / 2) *
                (image_height / image_width)

        Args:
            image_height:
                Height of the image in pixels.

            image_width:
                Width of the image in pixels.

        Returns:
            Vertical field of view in degrees.
        """

        horizontal_rad = math.radians(
            self.horizontal_fov / 2
        )

        vertical_rad = math.atan(
            math.tan(horizontal_rad) *
            (image_height / image_width)
        )

        vertical_fov = (
            2 * math.degrees(vertical_rad)
        )

        return vertical_fov


    def calculate_distance(
        self,
        detection,
        image_height,
        image_width,
        player_height=None
    ):
        """
        Calculate the distance between the player and the camera.

        Args:
            detection:
                Dictionary returned by detector.py

                Example:
                {
                    "x1": 500,
                    "y1": 200,
                    "x2": 700,
                    "y2": 800
                }

            image_height:
                Height of the image in pixels.

            image_width:
                Width of the image in pixels.

            player_height:
                Real height of the player in meters.
                Defaults to the height given in __init__.

        Returns:
            Distance in meters, or None when the player's
            full height is not visible (box truncated by the
            top or bottom edge of the frame).
        """

        if player_height is None:

            player_height = self.player_height


        if self.is_truncated(
            detection,
            image_height
        ):

            # Player is cut off -> height ratio is wrong,
            # so no reliable distance can be computed.
            return None


        box_height = (
            detection["y2"] - detection["y1"]
        )

        if box_height <= 0:

            raise ValueError(
                "Detection box has no vertical extent"
            )


        # Ratio of the player's height in the image

        ratio = height_ratio(
            box_height,
            image_height
        )


        # Vertical FOV from the frame aspect ratio

        vertical_fov = (
            self.calculate_vertical_fov(
                image_height,
                image_width
            )
        )


        # Distance from the pinhole camera formula

        distance = distance_from_ratio(
            player_height,
            vertical_fov,
            ratio
        )

        return distance
