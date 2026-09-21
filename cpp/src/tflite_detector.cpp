#include "tflite_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace tp {

namespace {

/**
 * Resize an image to a square of `size` keeping the aspect ratio.
 * Mirrors letterbox() in src/tflite_detector.py (fill value 114).
 */
cv::Mat letterbox(const cv::Mat& image, int size, double& scale,
                  int& left, int& top) {
    const int width = image.cols;
    const int height = image.rows;

    scale = std::min(static_cast<double>(size) / width,
                     static_cast<double>(size) / height);

    const int new_width = std::max(static_cast<int>(std::round(width * scale)), 1);
    const int new_height = std::max(static_cast<int>(std::round(height * scale)), 1);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height), 0, 0,
               cv::INTER_LINEAR);

    left = (size - new_width) / 2;
    top = (size - new_height) / 2;

    cv::Mat canvas(size, size, image.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(left, top, new_width, new_height)));
    return canvas;
}

#ifdef TP_HAVE_TFLITE
std::vector<float> tensorToFloat(const TfLiteTensor* tensor) {
    int count = 1;
    for (int i = 0; i < tensor->dims->size; ++i) {
        count *= tensor->dims->data[i];
    }

    std::vector<float> out(static_cast<std::size_t>(count));

    if (tensor->type == kTfLiteFloat32) {
        const float* data = tensor->data.f;
        out.assign(data, data + count);
    } else if (tensor->type == kTfLiteUInt8) {
        const uint8_t* data = tensor->data.uint8;
        const float scale = tensor->params.scale;
        const float zero_point = static_cast<float>(tensor->params.zero_point);
        for (int i = 0; i < count; ++i) {
            out[static_cast<std::size_t>(i)] = (static_cast<float>(data[i]) - zero_point) * scale;
        }
    } else if (tensor->type == kTfLiteInt8) {
        const int8_t* data = tensor->data.int8;
        const float scale = tensor->params.scale;
        const float zero_point = static_cast<float>(tensor->params.zero_point);
        for (int i = 0; i < count; ++i) {
            out[static_cast<std::size_t>(i)] = (static_cast<float>(data[i]) - zero_point) * scale;
        }
    } else if (tensor->type == kTfLiteInt32) {
        const int32_t* data = tensor->data.i32;
        for (int i = 0; i < count; ++i) {
            out[static_cast<std::size_t>(i)] = static_cast<float>(data[i]);
        }
    }

    return out;
}
#endif  // TP_HAVE_TFLITE

}  // namespace

TFLiteDetector::TFLiteDetector(const std::string& model_path, int threads)
    : model_path_(model_path) {
#ifdef TP_HAVE_TFLITE
    if (!std::filesystem::exists(model_path_)) {
        throw std::runtime_error("Model file not found: " + model_path_);
    }

    model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
    if (!model_) {
        throw std::runtime_error("Cannot build TFLite model from file: " + model_path_);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    if (builder(&interpreter_) != kTfLiteOk || !interpreter_) {
        throw std::runtime_error("Cannot build TFLite interpreter for: " + model_path_);
    }

    if (threads > 0) {
        interpreter_->SetNumThreads(threads);
    }
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Cannot allocate TFLite tensors for: " + model_path_);
    }

    const int input_index = interpreter_->inputs()[0];
    const TfLiteTensor* input = interpreter_->tensor(input_index);
    const int* dims = input->dims->data;

    // Fixed square inference size ([1, size, size, 3]).
    inference_size_ = dims[1];
    scale_bytes_ = (input->type == kTfLiteUInt8) || (input->type == kTfLiteInt8);

    prepareOutputDetails();

    std::cout << "TFLite model loaded: "
              << std::filesystem::path(model_path_).filename().string()
              << std::endl;
    std::cout << "Input shape: [";
    for (int i = 0; i < input->dims->size; ++i) {
        std::cout << (i == 0 ? "" : ", ") << dims[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "Input type: "
              << (scale_bytes_ ? "int8 (quantized)" : "float32")
              << std::endl;
#else
    (void)threads;
    throw std::runtime_error(
        "TensorFlow Lite support not compiled in (TP_HAVE_TFLITE). Rebuild "
        "with TP_ENABLE_TFLITE=ON and a working TensorFlow Lite install.");
#endif
}

void TFLiteDetector::prepareOutputDetails() {
#ifdef TP_HAVE_TFLITE
    yolo_head_ = false;
    yolo_index_ = 0;
    boxes_index_ = 0;
    classes_index_ = 1;
    scores_index_ = 2;
    num_index_ = -1;

    const int num_outputs = static_cast<int>(interpreter_->outputs().size());
    if (num_outputs > 3) {
        num_index_ = 3;
    }

    for (int index = 0; index < num_outputs; ++index) {
        const TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[index]);
        const int rank = output->dims->size;

        // YOLO head: [1, 84, N].
        if (rank == 3 && output->dims->data[1] == 84) {
            yolo_head_ = true;
            yolo_index_ = index;
            return;
        }

        // Post-processed head: boxes end with a 4.
        if (rank == 3 && output->dims->data[2] == 4) {
            boxes_index_ = index;
        }
    }
#else
    yolo_head_ = false;
#endif
}

void TFLiteDetector::preprocess(const cv::Mat& frame, double& scale,
                                int& left, int& top) {
#ifdef TP_HAVE_TFLITE
    // Models are trained on RGB images.
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    cv::Mat canvas = letterbox(rgb, inference_size_, scale, left, top);

    const TfLiteTensor* input = interpreter_->tensor(interpreter_->inputs()[0]);
    const std::size_t count = static_cast<std::size_t>(inference_size_) *
                              inference_size_ * 3;
    const uint8_t* src = canvas.data;

    if (scale_bytes_) {
        if (input->type == kTfLiteInt8) {
            int8_t* data = input->data.int8;
            for (std::size_t i = 0; i < count; ++i) {
                data[i] = static_cast<int8_t>(static_cast<int>(src[i]) - 128);
            }
        } else {
            uint8_t* data = input->data.uint8;
            std::memcpy(data, src, count);
        }
    } else {
        float* data = input->data.f;
        const float inv = 1.0f / 255.0f;
        for (std::size_t i = 0; i < count; ++i) {
            data[i] = src[i] * inv;
        }
    }
#endif
}

bool TFLiteDetector::invoke() {
#ifdef TP_HAVE_TFLITE
    return interpreter_->Invoke() == kTfLiteOk;
#else
    return false;
#endif
}

Detection TFLiteDetector::detectPerson(const cv::Mat& frame,
                                       float confidence_threshold) {
#ifdef TP_HAVE_TFLITE
    if (!frame.data) {
        return Detection{};
    }

    double scale;
    int left = 0, top = 0;
    preprocess(frame, scale, left, top);

    if (!invoke()) {
        return Detection{};
    }

    if (yolo_head_) {
        return parseYolo(scale, left, top, confidence_threshold);
    }
    return parsePostprocessed(scale, left, top, confidence_threshold);
#else
    (void)frame;
    (void)confidence_threshold;
    return Detection{};
#endif
}

Detection TFLiteDetector::mapBox(double x1, double y1, double x2, double y2,
                                 double scale, int left, int top) {
    Detection detection;
    detection.x1 = static_cast<int>((x1 - left) / scale);
    detection.y1 = static_cast<int>((y1 - top) / scale);
    detection.x2 = static_cast<int>((x2 - left) / scale);
    detection.y2 = static_cast<int>((y2 - top) / scale);
    return detection;
}

Detection TFLiteDetector::parseYolo(double scale, int left, int top,
                                    float confidence_threshold) {
#ifdef TP_HAVE_TFLITE
    const TfLiteTensor* tensor = interpreter_->tensor(interpreter_->outputs()[yolo_index_]);
    const int num_anchors = tensor->dims->data[2];
    const std::vector<float> predictions = tensorToFloat(tensor);

    Detection best;
    float best_confidence = confidence_threshold;

    for (int n = 0; n < num_anchors; ++n) {
        auto at = [&](int k) -> float {
            return predictions[static_cast<std::size_t>(k) * num_anchors + n];
        };

        const float x_center = at(0);
        const float y_center = at(1);
        const float width = at(2);
        const float height = at(3);
        const float confidence = at(4 + PERSON_CLASS_ID);

        if (confidence < best_confidence) {
            continue;
        }
        best_confidence = confidence;

        best = mapBox(x_center - width / 2, y_center - height / 2,
                      x_center + width / 2, y_center + height / 2,
                      scale, left, top);
        best.class_name = "person";
        best.confidence = confidence;
    }

    return best;
#else
    (void)scale;
    (void)left;
    (void)top;
    (void)confidence_threshold;
    return Detection{};
#endif
}

Detection TFLiteDetector::parsePostprocessed(double scale, int left, int top,
                                             float confidence_threshold) {
#ifdef TP_HAVE_TFLITE
    const std::vector<float> boxes =
        tensorToFloat(interpreter_->tensor(interpreter_->outputs()[boxes_index_]));
    const std::vector<float> scores =
        tensorToFloat(interpreter_->tensor(interpreter_->outputs()[scores_index_]));
    const std::vector<float> classes =
        tensorToFloat(interpreter_->tensor(interpreter_->outputs()[classes_index_]));

    const int max_detections = static_cast<int>(scores.size());
    int num_detections = max_detections;
    if (num_index_ >= 0) {
        const std::vector<float> counts =
            tensorToFloat(interpreter_->tensor(interpreter_->outputs()[num_index_]));
        const int detected = counts.empty() ? 0 : static_cast<int>(counts[0]);
        if (detected > 0) {
            num_detections = std::min(detected, max_detections);
        }
    }

    const int frame_size = inference_size_;
    for (int i = 0; i < num_detections; ++i) {
        const int class_id = static_cast<int>(classes[i]);
        const float confidence = scores[i];
        if (class_id != PERSON_CLASS_ID) {
            continue;
        }
        if (confidence < confidence_threshold) {
            continue;
        }

        const double ymin = boxes[i * 4 + 0];
        const double xmin = boxes[i * 4 + 1];
        const double ymax = boxes[i * 4 + 2];
        const double xmax = boxes[i * 4 + 3];

        Detection detection = mapBox(xmin * frame_size, ymin * frame_size,
                                     xmax * frame_size, ymax * frame_size,
                                     scale, left, top);
        detection.class_name = "person";
        detection.confidence = confidence;
        return detection;
    }

    return Detection{};
#else
    (void)scale;
    (void)left;
    (void)top;
    (void)confidence_threshold;
    return Detection{};
#endif
}

}  // namespace tp