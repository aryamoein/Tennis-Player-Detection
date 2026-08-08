import math


class VerticalAngleCalculator:
    """
    This class handles vertical angle calculations
    between the image center and a point in the image.
    """


    def __init__(self, horizontal_fov):
        """
        Args:
            horizontal_fov:
                Horizontal field of view of the camera in degrees.

        The vertical field of view is derived from the
        horizontal field of view and the frame aspect ratio.
        """

        self.horizontal_fov = horizontal_fov

        self.vertical_fov = None


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


    def calculate_vertical_angle(
        self,
        head_y,
        image_center_y,
        image_height,
        image_width
    ):
        """
        Calculate the vertical angle between the
        image center and the head position.

        Args:
            head_y:
                Head y coordinate (top of the detection box).

            image_center_y:
                Center y coordinate of the image.

            image_height:
                Height of the image.

            image_width:
                Width of the image.

        Returns:
            Angle in degrees.

            Positive:
                Head is above the image center.

            Negative:
                Head is below the image center.
        """

        if self.vertical_fov is None:

            self.vertical_fov = (
                self.calculate_vertical_fov(
                    image_height,
                    image_width
                )
            )


        # Difference between head and image center

        pixel_offset = (
            image_center_y - head_y
        )


        # How many degrees correspond to one pixel

        degree_per_pixel = (
            self.vertical_fov / image_height
        )


        # Convert pixel difference to angle

        angle = (
            pixel_offset * degree_per_pixel
        )


        return angle