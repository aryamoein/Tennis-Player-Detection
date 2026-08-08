class CameraGeometry:
    """
    This class handles geometric calculations
    between image coordinates and camera angles.
    """


    def __init__(self, horizontal_fov):
        """
        Args:
            horizontal_fov:
                Horizontal field of view of the camera in degrees.

        Example:
            A wide camera may have around 90 degrees FOV.
        """

        self.horizontal_fov = horizontal_fov



    def get_player_center(self, detection):
        """
        Calculate the center point of the detected player.

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

        Returns:
            (x, y) center coordinate of player
        """


        x_center = (
            detection["x1"] + detection["x2"]
        ) / 2


        y_center = (
            detection["y1"] + detection["y2"]
        ) / 2


        return int(x_center), int(y_center)



    def get_image_center(self, frame):
        """
        Calculate the center point of the image.

        Args:
            frame:
                OpenCV image

        Returns:
            (x, y) center of image
        """


        height, width = frame.shape[:2]


        x_center = width / 2

        y_center = height / 2


        return int(x_center), int(y_center)



    def calculate_horizontal_angle(
        self,
        player_x,
        image_center_x,
        image_width
    ):
        """
        Calculate camera rotation angle.

        Args:
            player_x:
                Player x coordinate

            image_center_x:
                Center x coordinate of image

            image_width:
                Width of image

        Returns:
            Angle in degrees

            Positive:
                Player is on the right side

            Negative:
                Player is on the left side
        """


        # Difference between player and image center

        pixel_offset = (
            player_x - image_center_x
        )


        # How many degrees correspond to one pixel

        degree_per_pixel = (
            self.horizontal_fov / image_width
        )


        # Convert pixel difference to angle

        angle = (
            pixel_offset * degree_per_pixel
        )


        return angle