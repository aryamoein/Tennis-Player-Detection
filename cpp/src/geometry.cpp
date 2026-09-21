#include "geometry.hpp"

namespace tp {

cv::Point CameraGeometry::getPlayerCenter(const Detection& detection) const {
    return cv::Point((detection.x1 + detection.x2) / 2,
                     (detection.y1 + detection.y2) / 2);
}

cv::Point CameraGeometry::getImageCenter(const cv::Mat& frame) const {
    return cv::Point(frame.cols / 2, frame.rows / 2);
}

double CameraGeometry::calculateHorizontalAngle(double player_x,
                                                double image_center_x,
                                                double image_width) const {
    const double pixel_offset = player_x - image_center_x;
    const double degree_per_pixel = horizontal_fov_ / image_width;
    return pixel_offset * degree_per_pixel;
}

}  // namespace tp