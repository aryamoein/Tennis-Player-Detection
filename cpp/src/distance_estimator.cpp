#include "distance_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "config.hpp"
#include "distance_formula.hpp"

namespace tp {

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

DistanceEstimator::DistanceEstimator(double horizontal_fov,
                                     double player_height,
                                     std::string face_model)
    : horizontal_fov_(horizontal_fov),
      player_height_(player_height),
      face_model_(std::move(face_model)) {}

cv::Ptr<cv::FaceDetectorYN> DistanceEstimator::loadFaceDetector() {
    if (!face_detector_.empty()) {
        return face_detector_;
    }

    // Fall back to the bundled model next to the project.
    std::string model = face_model_;
    if (model.empty()) {
        model = (Config::projectDirectory() / "models" /
                 "yunet_face_detection.onnx").string();
    }
    if (!std::filesystem::exists(model)) {
        return cv::Ptr<cv::FaceDetectorYN>();
    }

    // Input size is updated per frame in detectHeadHeight, so any
    // placeholder is fine here.
    try {
        face_detector_ = cv::FaceDetectorYN::create(model, "", cv::Size(320, 320));
    } catch (const cv::Exception&) {
        return cv::Ptr<cv::FaceDetectorYN>();
    }
    return face_detector_;
}

double DistanceEstimator::calculateVerticalFov(int image_height,
                                               int image_width) const {
    const double horizontal_rad = (horizontal_fov_ / 2.0) * (kPi / 180.0);
    const double vertical_rad = std::atan(
        std::tan(horizontal_rad) *
        (static_cast<double>(image_height) / image_width));
    return 2.0 * vertical_rad * (180.0 / kPi);
}

bool DistanceEstimator::isTruncated(const Detection& detection,
                                    int image_height, int tolerance) const {
    return detection.y1 <= tolerance ||
           detection.y2 >= image_height - 1 - tolerance;
}

std::optional<double> DistanceEstimator::estimateFromFullHeight(
    const Detection& detection, int image_height, int image_width) const {
    if (isTruncated(detection, image_height)) {
        return std::nullopt;
    }

    const int box_height = detection.y2 - detection.y1;
    if (box_height <= 0) {
        return std::nullopt;
    }

    const double ratio = heightRatio(box_height, image_height);
    const double vertical_fov = calculateVerticalFov(image_height, image_width);
    return distanceFromRatio(player_height_, vertical_fov, ratio);
}

std::optional<int> DistanceEstimator::detectHeadHeight(
    const cv::Mat& frame, const Detection& detection) {
    const cv::Ptr<cv::FaceDetectorYN> detector = loadFaceDetector();
    if (detector.empty()) {
        return std::nullopt;
    }

    const int x1 = std::max(detection.x1, 0);
    const int y1 = std::max(detection.y1, 0);
    const int x2 = detection.x2;
    const int y2 = detection.y2;

    // Look for the head in the top half of the box.
    const int head_bottom = y1 + (y2 - y1) / 2;
    if (head_bottom <= y1) {
        return std::nullopt;
    }

    // FaceDetectorYN input size must match the frame.
    detector->setInputSize(cv::Size(frame.cols, frame.rows));

    cv::Mat faces;
    detector->detect(frame, faces);
    if (faces.empty()) {
        return std::nullopt;
    }

    int best = -1;
    int best_area = 0;
    for (int i = 0; i < faces.rows; ++i) {
        const int fx1 = static_cast<int>(faces.at<float>(i, 0));
        const int fy1 = static_cast<int>(faces.at<float>(i, 1));
        const int face_width = static_cast<int>(faces.at<float>(i, 2));
        const int face_height = static_cast<int>(faces.at<float>(i, 3));

        // Keep only faces inside the head region.
        if (fx1 < x1 || fy1 < y1 || fx1 + face_width > x2 ||
            fy1 + face_height > head_bottom) {
            continue;
        }

        const int area = face_width * face_height;
        if (area > best_area) {
            best_area = area;
            best = face_height;
        }
    }

    if (best <= 0) {
        return std::nullopt;
    }
    return best;
}

std::optional<double> DistanceEstimator::estimateFromHead(
    const cv::Mat& frame, const Detection& detection,
    int image_height, int image_width) {
    const std::optional<int> head_px = detectHeadHeight(frame, detection);
    if (!head_px.has_value() || *head_px <= 0) {
        return std::nullopt;
    }

    const double real_head = player_height_ * HEAD_RATIO;
    const double ratio = heightRatio(static_cast<double>(*head_px),
                                     static_cast<double>(image_height));
    const double vertical_fov = calculateVerticalFov(image_height, image_width);
    return distanceFromRatio(real_head, vertical_fov, ratio);
}

std::pair<std::optional<double>, std::map<std::string, double>>
DistanceEstimator::estimate(const cv::Mat& frame,
                            const Detection& detection,
                            int image_height, int image_width) {
    std::map<std::string, double> estimates;

    const std::optional<double> full = estimateFromFullHeight(
        detection, image_height, image_width);
    if (full.has_value()) {
        estimates["full_height"] = *full;
        return {full, estimates};
    }

    const std::optional<double> head = estimateFromHead(
        frame, detection, image_height, image_width);
    if (head.has_value()) {
        estimates["head"] = *head;
        return {head, estimates};
    }

    return {std::nullopt, estimates};
}

}  // namespace tp