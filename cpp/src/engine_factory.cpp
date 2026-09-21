#include "engine_factory.hpp"

#include <iostream>

#include "tflite_detector.hpp"

namespace tp {

namespace {

std::unique_ptr<Detector> loadTflite(const fs::path& path, int threads) {
    try {
        auto detector = std::make_unique<TFLiteDetector>(path.string(), threads);
        std::cout << "Using TFLite engine: "
                  << path.filename().string() << std::endl;
        return detector;
    } catch (const std::exception& error) {
        std::cout << "TFLite unavailable (" << error.what()
                  << "); falling back to ONNX." << std::endl;
        return nullptr;
    }
}

std::unique_ptr<Detector> loadOnnx(const fs::path& path, int inference_size) {
    std::cout << "Using ONNX engine: "
              << path.filename().string() << std::endl;
    return std::make_unique<YOLODetector>(path.string(), inference_size);
}

}  // namespace

std::unique_ptr<Detector> createDetector(
    const fs::path& project_dir, const Config& cfg,
    const std::optional<std::string>& model_override,
    const std::optional<int>& threads_override) {
    const int threads = threads_override.value_or(cfg.threads);

    if (model_override.has_value()) {
        // Explicit model requested.
        const fs::path path = project_dir / "models" / *model_override;
        if (path.extension() == ".tflite") {
            if (auto detector = loadTflite(path, threads)) {
                return detector;
            }
        }
        return loadOnnx(path, cfg.inference_size);
    }

    // Default model from config (TFLite preferred).
    const fs::path config_path = project_dir / cfg.model_file;
    if (config_path.extension() == ".tflite") {
        if (auto detector = loadTflite(config_path, threads)) {
            return detector;
        }
    }

    const fs::path onnx_path = project_dir / "models" / "yolov8n_320.onnx";
    return loadOnnx(onnx_path, cfg.inference_size);
}

}  // namespace tp