#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "detector.hpp"

#ifdef TP_HAVE_TFLITE
#include <memory>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#endif

namespace tp {

/**
 * Person detection using a TensorFlow Lite model.
 *
 * Mirrors src/tflite_detector.py (TFLiteDetector). Supports
 * YOLO-style heads (single raw output tensor) and post-processed
 * heads with built-in NMS (boxes, classes, scores, num_detections).
 *
 * Provides the same detectPerson() interface as YOLODetector.
 */
class TFLiteDetector final : public Detector {
public:
    static constexpr int PERSON_CLASS_ID = 0;

    explicit TFLiteDetector(const std::string& model_path, int threads = 4);

    Detection detectPerson(const cv::Mat& frame,
                           float confidence_threshold = 0.5f) override;

    int inferenceSize() const { return inference_size_; }

private:
    void preprocess(const cv::Mat& frame, double& scale, int& left, int& top);
    bool invoke();
    Detection parseYolo(double scale, int left, int top,
                        float confidence_threshold);
    Detection parsePostprocessed(double scale, int left, int top,
                                 float confidence_threshold);
    static Detection mapBox(double x1, double y1, double x2, double y2,
                            double scale, int left, int top);
    void prepareOutputDetails();

    std::string model_path_;
    int inference_size_ = 0;
    bool scale_bytes_ = true;
    bool yolo_head_ = false;
    int yolo_index_ = 0;
    int boxes_index_ = 0;
    int classes_index_ = 1;
    int scores_index_ = 2;
    int num_index_ = -1;

#ifdef TP_HAVE_TFLITE
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
#endif
};

}  // namespace tp