import math


def height_ratio(
    player_height_px,
    image_height_px
):
    """
    Fraction of the image height occupied by the player.

    Args:
        player_height_px:
            Height of the detected player in pixels.

        image_height_px:
            Height of the frame in pixels.

    Returns:
        Ratio X in (0, 1].
    """

    if image_height_px <= 0:

        raise ValueError(
            "image_height_px must be positive"
        )

    if player_height_px < 0:

        raise ValueError(
            "player_height_px must be non-negative"
        )

    return player_height_px / image_height_px


def distance_from_ratio(
    player_height,
    vertical_fov,
    height_ratio_value
):
    """
    Distance between the player and the camera in meters.

    Uses the pinhole camera distance formula:

        x = L / (2 * X * tan(theta / 2))

    Where:
        x = distance between the player and the camera
        L = player's real height (meters)
        theta = vertical field of view of the camera (degrees)
        X = ratio of the player's height in the image (0..1)

    Args:
        player_height:
            Real height of the player in meters.

        vertical_fov:
            Vertical field of view of the camera in degrees.

        height_ratio_value:
            Ratio of the player's height in the image (0..1).

    Returns:
        Distance in meters.
    """

    if vertical_fov <= 0 or vertical_fov >= 180:

        raise ValueError(
            "vertical_fov must be between 0 and 180 degrees"
        )

    if height_ratio_value <= 0:

        raise ValueError(
            "height_ratio_value must be positive"
        )

    half_angle = math.radians(vertical_fov / 2.0)

    distance = (
        player_height
        / (
            2.0
            * height_ratio_value
            * math.tan(half_angle)
        )
    )

    return distance


def distance_from_width_ratio(
    real_width,
    horizontal_fov,
    width_ratio_value
):
    """
    Distance between an object and the camera in meters,
    computed from a known width instead of a height.

    Uses the pinhole camera distance formula:

        x = W / (2 * X * tan(theta / 2))

    Where:
        x = distance between the object and the camera
        W = object's real width (meters)
        theta = horizontal field of view of the camera (degrees)
        X = ratio of the object's width in the image (0..1)

    Args:
        real_width:
            Real width of the object in meters.

        horizontal_fov:
            Horizontal field of view of the camera in degrees.

        width_ratio_value:
            Ratio of the object's width in the image (0..1).

    Returns:
        Distance in meters.
    """

    if horizontal_fov <= 0 or horizontal_fov >= 180:

        raise ValueError(
            "horizontal_fov must be between 0 and 180 degrees"
        )

    if width_ratio_value <= 0:

        raise ValueError(
            "width_ratio_value must be positive"
        )

    half_angle = math.radians(horizontal_fov / 2.0)

    distance = (
        real_width
        / (
            2.0
            * width_ratio_value
            * math.tan(half_angle)
        )
    )

    return distance
