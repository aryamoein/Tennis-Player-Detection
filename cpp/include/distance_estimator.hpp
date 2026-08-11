#pragma once

#include <map>
#include <optional>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/objdetect/face_detect_yunet.hpp>

#include "detector.hpp"

namespace tp {

/**
 * Estimates the player-camera distance.
 *
 * Mirrors src/distance_estimator.py (DistanceEstimator):
 *   1. Full body height  - whole detection box height (preferred;
 *                          position-invariant under pinhole projection).
 *   2. Upper body (head) - head detected with YuNet inside the top of
 *                          the box (fallback when truncated).
 */
class DistanceEstimator {
public:
    // Average head-to-body ratio (~7.5 heads tall).
    static constexpr double HEAD_RATIO = 1.0 / 7.5;

    DistanceEstimator(double horizontal_fov,
                      double player_height = 1.75,
                      std::string face_model = "");

    /** Derive vertical FOV from the horizontal FOV and frame dims. */
    double calculateVerticalFov(int image_height, int image_width) const;

    /** True when the box is cut off by the top or bottom edge. */
    bool isTruncated(const Detection& detection, int image_height,
                     int tolerance = 2) const;

    /** Distance from the whole box height (None when truncated). */
    std::optional<double> estimateFromFullHeight(const Detection& detection,
                                                 int image_height,
                                                 int image_width) const;

    /** Head height in pixels, or nullopt when no face is found. */
    std::optional<int> detectHeadHeight(const cv::Mat& frame,
                                        const Detection& detection);

    /** Distance from the head height (upper body). */
    std::optional<double> estimateFromHead(const cv::Mat& frame,
                                           const Detection& detection,
                                           int image_height,
                                           int image_width);

    /**
     * Estimate the player-camera distance.
     *
     * Returns (combined_distance, estimates) where estimates maps
     * approach name -> distance for comparison. combined_distance is
     * nullopt when nothing is usable.
     */
    std::pair<std::optional<double>, std::map<std::string, double>> estimate(
        const cv::Mat& frame, const Detection& detection,
        int image_height, int image_width);

private:
    cv::Ptr<cv::FaceDetectorYN> loadFaceDetector();

    double horizontal_fov_;
    double player_height_;
    std::string face_model_;
    cv::Ptr<cv::FaceDetectorYN> face_detector_;
};

}  // namespace tp