#pragma once

namespace tp {

/**
 * Fraction of the image height occupied by the player.
 *
 * Mirrors height_ratio() in src/distance_formula.py.
 *
 * Returns a ratio X in (0, 1].
 */
double heightRatio(double player_height_px, double image_height_px);

/**
 * Distance between the player and the camera in meters.
 *
 * Uses the pinhole camera distance formula:
 *     x = L / (2 * X * tan(theta / 2))
 *
 * x      = distance between the player and the camera
 * L      = player's real height (meters)
 * theta  = vertical field of view of the camera (degrees)
 * X      = ratio of the player's height in the image (0..1)
 *
 * Mirrors distance_from_ratio() in src/distance_formula.py.
 */
double distanceFromRatio(double player_height,
                         double vertical_fov,
                         double height_ratio_value);

}  // namespace tp