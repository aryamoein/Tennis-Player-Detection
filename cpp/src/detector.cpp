#include "detector.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

#ifdef TP_HAVE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace tp {

#ifdef TP_HAVE_ONNX

/** Holds all ONNX Runtime state out of the public header. */
class YOLODetector::Impl {
public:
    Ort::Env env;
    Ort::SessionOptions options;
    Ort::Session session;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;

    int channel_count = 3;

    Impl()
        : env(ORT_LOGGING_LEVEL_WARNING, "tennis"),
          session(nullptr) {}

    // Ort::Session has no default constructor, so build it explicitly.
    void build(const std::string& path) {
        session = Ort::Session(env, path.c_str(), options);
    }
};

#else  // !TP_HAVE_ONNX

/** Empty placeholder so `sizeof(YOLODetector)` stays identical across TUs. */
class YOLODetector::Impl {};

#endif  // TP_HAVE_ONNX

YOLODetector::YOLODetector(const std::string& model_path,
                           int inference_size,
                           int threads)
    : model_path_(model_path),
      inference_size_(inference_size),
      threads_(threads) {
#ifdef TP_HAVE_ONNX
    impl_ = std::make_unique<Impl>();
    try {
        impl_->options.SetIntraOpNumThreads(std::max(1, threads_));
        impl_->options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        impl_->build(model_path_);

        std::cout << "YOLO model loaded successfully" << std::endl;

        Ort::AllocatorWithDefaultOptions allocator;

        // Input information.
        std::vector<int64_t> input_shape;
        {
            auto type_info = impl_->session.GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shape = tensor_info.GetShape();
            impl_->input_names.emplace_back(
                impl_->session.GetInputNameAllocated(0, allocator).get());
        }

        // Use the model's own resolution when it declares one; the
        // constructor argument (Config.INFERENCE_SIZE) is the fallback.
        if (input_shape.size() >= 4 && input_shape[2] > 0) {
            inference_size_ = static_cast<int>(input_shape[2]);
        }

        // Output information.
        const size_t num_outputs = impl_->session.GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            impl_->output_names.emplace_back(
                impl_->session.GetOutputNameAllocated(i, allocator).get());
        }

        impl_->input_name_ptrs.reserve(impl_->input_names.size());
        for (const auto& name : impl_->input_names) {
            impl_->input_name_ptrs.push_back(name.c_str());
        }
        impl_->output_name_ptrs.reserve(impl_->output_names.size());
        for (const auto& name : impl_->output_names) {
            impl_->output_name_ptrs.push_back(name.c_str());
        }

        std::cout << "Input name: " << impl_->input_names[0] << std::endl;
        std::cout << "Input shape: [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << (i == 0 ? "" : ", ") << input_shape[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "Number of outputs: " << num_outputs << std::endl;
        for (size_t i = 0; i < num_outputs; ++i) {
            // Keep the Ort::TypeInfo alive while its (unowned) tensor shape
            // view is in use; otherwise the view dangles and dims read as
            // garbage. (GetTensorTypeAndShapeInfo returns an unowned view into
            // the TypeInfo, so the temporary in
            // `GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo()` is a
            // use-after-free.)
            Ort::TypeInfo output_info = impl_->session.GetOutputTypeInfo(i);
            auto shape = output_info.GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "Output: " << impl_->output_names[i] << " [";
            for (size_t j = 0; j < shape.size(); ++j) {
                std::cout << (j == 0 ? "" : ", ");
                if (shape[j] < 0) {
                    std::cout << "?";  // dynamic dimension
                } else {
                    std::cout << shape[j];
                }
            }
            std::cout << "]" << std::endl;
        }
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "Failed to load ONNX model '" + model_path_ + "': " + error.what());
    }
#else
    (void)inference_size;
    (void)threads;
    throw std::runtime_error(
        "ONNX Runtime support not compiled in (TP_HAVE_ONNX). Rebuild with "
        "TP_ENABLE_ONNX=ON and a working onnxruntime install.");
#endif
}

YOLODetector::~YOLODetector() = default;

Detection YOLODetector::detectPerson(const cv::Mat& frame,
                                     float confidence_threshold) {
#ifdef TP_HAVE_ONNX
    if (!impl_ || !frame.data) {
        return Detection{};
    }

    const int size = inference_size_;
    const int plane = size * size;
    const int channels = impl_->channel_count;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);

    // BGR -> RGB, HWC -> CHW, normalized to [0, 1].
    std::vector<float> input(static_cast<std::size_t>(channels) * plane);
    const float inv = 1.0f / 255.0f;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const cv::Vec3b& pixel = resized.at<cv::Vec3b>(y, x);
            const int offset = y * size + x;
            input[offset] = pixel[2] * inv;                    // R
            input[plane + offset] = pixel[1] * inv;            // G
            input[2 * plane + offset] = pixel[0] * inv;        // B
        }
    }

    const std::vector<int64_t> shape = {1, channels, size, size};
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input.data(), input.size(), shape.data(), shape.size());

    auto outputs = impl_->session.Run(
        Ort::RunOptions(),
        impl_->input_name_ptrs.data(),
        &input_tensor,
        1,
        impl_->output_name_ptrs.data(),
        impl_->output_name_ptrs.size());

    if (outputs.empty()) {
        return Detection{};
    }

    Ort::Value& output = outputs[0];
    const auto out_info = output.GetTensorTypeAndShapeInfo();
    const auto out_shape = out_info.GetShape();
    const float* data = output.GetTensorData<float>();

    // Expect a YOLO-style head: [1, 84, N].
    if (out_shape.size() != 3) {
        return Detection{};
    }
    const int channels_out = static_cast<int>(out_shape[1]);
    const int num_anchors = static_cast<int>(out_shape[2]);
    if (channels_out <= 4 + PERSON_CLASS_ID) {
        return Detection{};
    }

    Detection best;
    float best_confidence = confidence_threshold;

    for (int n = 0; n < num_anchors; ++n) {
        auto at = [&](int k) -> float {
            return data[static_cast<std::size_t>(k) * num_anchors + n];
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

        const double scale_x = static_cast<double>(frame.cols) / size;
        const double scale_y = static_cast<double>(frame.rows) / size;

        best = Detection{};
        best.class_name = "person";
        best.confidence = confidence;
        best.x1 = static_cast<int>((x_center - width / 2) * scale_x);
        best.y1 = static_cast<int>((y_center - height / 2) * scale_y);
        best.x2 = static_cast<int>((x_center + width / 2) * scale_x);
        best.y2 = static_cast<int>((y_center + height / 2) * scale_y);
    }

    return best;
#else
    (void)frame;
    (void)confidence_threshold;
    return Detection{};
#endif
}

}  // namespace tp