#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "config.hpp"
#include "detector.hpp"

namespace tp {

/**
 * Load the configured detection model.
 *
 * Mirrors create_detector() in src/main.py: prefers TensorFlow Lite
 * and falls back to ONNX when TFLite is unavailable.
 */
std::unique_ptr<Detector> createDetector(
    const fs::path& project_dir, const Config& cfg,
    const std::optional<std::string>& model_override,
    const std::optional<int>& threads_override);

}  // namespace tp