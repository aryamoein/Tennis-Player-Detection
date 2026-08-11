#include "distance_formula.hpp"

#include <cmath>
#include <stdexcept>

namespace tp {

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

double heightRatio(double player_height_px, double image_height_px) {
    if (image_height_px <= 0) {
        throw std::invalid_argument("image_height_px must be positive");
    }
    if (player_height_px < 0) {
        throw std::invalid_argument("player_height_px must be non-negative");
    }
    return player_height_px / image_height_px;
}

double distanceFromRatio(double player_height,
                         double vertical_fov,
                         double height_ratio_value) {
    if (vertical_fov <= 0 || vertical_fov >= 180) {
        throw std::invalid_argument("vertical_fov must be between 0 and 180 degrees");
    }
    if (height_ratio_value <= 0) {
        throw std::invalid_argument("height_ratio_value must be positive");
    }

    const double half_angle = (vertical_fov / 2.0) * (kPi / 180.0);
    return player_height /
           (2.0 * height_ratio_value * std::tan(half_angle));
}

}  // namespace tp