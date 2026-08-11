#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

namespace tp {

/**
 * Person detection result. Coordinates are in the original frame.
 *
 * Mirrors the dict returned by detector.py / tflite_detector.py.
 */
struct Detection {
    std::string class_name;
    float confidence = 0.0f;
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;

    /** False when no detection was found (mirrors Python `None`). */
    bool valid() const { return confidence > 0.0f; }
};

/** Common interface shared by the ONNX and TFLite backends. */
class Detector {
public:
    virtual ~Detector() = default;

    virtual Detection detectPerson(const cv::Mat& frame,
                                   float confidence_threshold = 0.5f) = 0;
};

/**
 * Person detection with a YOLOv8 ONNX model.
 *
 * Mirrors src/detector.py (YOLODetector). Input is resized
 * (not letterboxed) to a square and normalized to CHW float.
 */
class YOLODetector final : public Detector {
public:
    static constexpr int PERSON_CLASS_ID = 0;

    explicit YOLODetector(const std::string& model_path,
                          int inference_size = 640,
                          int threads = 4);

    ~YOLODetector() override;

    Detection detectPerson(const cv::Mat& frame,
                           float confidence_threshold = 0.5f) override;

    int inferenceSize() const { return inference_size_; }

private:
    std::string model_path_;
    int inference_size_ = 640;
    int threads_ = 4;

    // Always declared so sizeof(YOLODetector) is identical for every
    // translation unit regardless of which backend macros are defined.
    // Keeping the ONNX Runtime state out of the public header avoids
    // layout surprises between tp_core and tp_detectors.
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tp