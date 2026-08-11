#pragma once

#include <opencv2/core.hpp>

#include "detector.hpp"

namespace tp {

/**
 * Geometric calculations between image coordinates and camera angles.
 *
 * Mirrors src/geometry.py (CameraGeometry).
 */
class CameraGeometry {
public:
    explicit CameraGeometry(double horizontal_fov)
        : horizontal_fov_(horizontal_fov) {}

    double horizontalFov() const { return horizontal_fov_; }

    /** Center point of the detected player box (truncated to int). */
    cv::Point getPlayerCenter(const Detection& detection) const;

    /** Center point of the image. */
    cv::Point getImageCenter(const cv::Mat& frame) const;

    /**
     * Camera rotation angle (horizontal / pan) in degrees.
     *
     * Positive: player is on the right side.
     * Negative: player is on the left side.
     */
    double calculateHorizontalAngle(double player_x,
                                    double image_center_x,
                                    double image_width) const;

private:
    double horizontal_fov_;
};

}  // namespace tp