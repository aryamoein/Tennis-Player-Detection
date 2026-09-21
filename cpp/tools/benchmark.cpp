// Benchmark detection models on the Raspberry Pi.
//
// Mirrors tools/benchmark.py: measures pure inference FPS on a
// synthetic frame, making model-to-model comparison fair and
// camera-independent.
//
// Usage (on the Pi):
//     ./benchmark --model models/ssd_mobilenet_v2_coco_int8_300.tflite
//     ./benchmark --model m1.tflite m2.tflite --frames 100

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "tflite_detector.hpp"

namespace {

namespace fs = std::filesystem;

struct Args {
    std::vector<std::string> models;
    int frames = 100;
    int threads = 4;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 < argc) {
                return argv[++i];
            }
            return std::string();
        };

        if (flag == "--frames") {
            args.frames = std::stoi(next());
        } else if (flag == "--threads") {
            args.threads = std::stoi(next());
        } else if (flag == "--model") {
            args.models.push_back(next());
        } else {
            // Bare positional arguments are treated as model paths.
            args.models.push_back(flag);
        }
    }
    return args;
}

/** Time inference on a batch of frames. Returns FPS (0 when skipped). */
double benchmark(const fs::path& model_path, int threads, int frames) {
    std::unique_ptr<tp::TFLiteDetector> detector;
    try {
        detector = std::make_unique<tp::TFLiteDetector>(model_path.string(), threads);
    } catch (const std::exception& error) {
        std::cout << std::left << std::setw(45)
                  << model_path.filename().string()
                  << " SKIP (TFLite unavailable: " << error.what() << ")\n";
        return 0.0;
    }

    // Synthetic 480p frame (typical camera resolution).
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    // Warm up.
    detector->detectPerson(frame, 0.5f);

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; ++i) {
        detector->detectPerson(frame, 0.5f);
    }
    const auto end = std::chrono::steady_clock::now();

    const double elapsed =
        std::chrono::duration<double>(end - start).count();
    const double ms = 1000.0 * elapsed / frames;
    const double fps = frames / elapsed;

    std::cout << std::left << std::setw(45)
              << model_path.filename().string() << " "
              << std::fixed << std::setprecision(1) << std::setw(8) << ms
              << " ms " << std::setprecision(2) << std::setw(6) << fps
              << " FPS\n";
    return fps;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);

    std::vector<std::string> models = args.models;
    if (models.empty()) {
        models = {
            "models/ssd_mobilenet_v2_coco_int8_300.tflite",
            "models/efficientdet_lite0_coco_int8_320.tflite",
        };
    }

    std::cout << "Benchmarking with " << args.threads
              << " thread(s), " << args.frames << " frames each.\n\n";

    std::string best_model;
    double best_fps = 0.0;

    for (const auto& model : models) {
        const fs::path path = model;
        if (!fs::exists(path)) {
            std::cout << "Skip (not found): " << model << "\n";
            continue;
        }

        const double fps = benchmark(path, args.threads, args.frames);
        if (fps > best_fps) {
            best_fps = fps;
            best_model = model;
        }
    }

    std::cout << "\nFastest model:\n";
    std::cout << "  " << best_model << " @ " << best_fps << " FPS\n";

    return 0;
}