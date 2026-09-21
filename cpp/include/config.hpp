#pragma once

#include <filesystem>
#include <string>

namespace tp {

namespace fs = std::filesystem;

/**
 * Central place for all tunable settings.
 *
 * Mirrors src/config.py, optimized for Raspberry Pi Zero 2 W
 * (64-bit OS).
 */
struct Config {
    // ------------------------------------------------------------
    // Video source
    // ------------------------------------------------------------
    int camera_index = 0;                    // USB camera index (0, 1, ...)
    std::string video_path = "videos/test.mp4";
    int capture_width = 640;
    int capture_height = 480;
    double capture_fps = 30.0;
    bool use_mjpeg = true;

    // ------------------------------------------------------------
    // Detection model
    // ------------------------------------------------------------
    std::string model_file = "models/yolov8n_320.onnx";  // default ONNX fallback
    int inference_size = 320;                // only the ONNX backend uses this
    int threads = 4;                         // Pi Zero 2 W has 4 cores
    double confidence_threshold = 0.5;
    int frame_skip = 0;                      // run inference every (frame_skip + 1)

    // ------------------------------------------------------------
    // Camera geometry
    // ------------------------------------------------------------
    // Samsung Galaxy S24 Ultra main camera (rear wide, 4:3):
    // 68.3 deg horizontal FOV -> ~54.0 deg vertical FOV at 4:3.
    double horizontal_fov = 68.3;

    // ------------------------------------------------------------
    // Player
    // ------------------------------------------------------------
    double player_height = 1.75;             // real height in meters (175 cm)

    // Root of the repository (the folder containing src/ and cpp/).
    static fs::path projectDirectory();

    fs::path modelPath() const;              // projectDirectory() / model_file
    fs::path videoPath() const;              // projectDirectory() / video_path
};

}  // namespace tp
