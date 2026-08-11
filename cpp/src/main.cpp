#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "capture.hpp"
#include "config.hpp"
#include "distance_estimator.hpp"
#include "engine_factory.hpp"
#include "geometry.hpp"

namespace tp {

namespace {

struct CliArgs {
    std::optional<int> camera;
    std::optional<std::string> file;
    std::optional<std::string> model;
    std::optional<int> threads;
    std::optional<int> skip;
    bool no_gui = false;
};

CliArgs parseArgs(int argc, char** argv) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 < argc) {
                return argv[++i];
            }
            return std::string();
        };

        if (flag == "--camera") {
            args.camera = std::stoi(next());
        } else if (flag == "--file") {
            args.file = next();
        } else if (flag == "--model") {
            args.model = next();
        } else if (flag == "--threads") {
            args.threads = std::stoi(next());
        } else if (flag == "--skip") {
            args.skip = std::stoi(next());
        } else if (flag == "--no-gui") {
            args.no_gui = true;
        } else {
            std::cerr << "Unknown argument: " << flag << std::endl;
        }
    }
    return args;
}

}  // namespace

}  // namespace tp

int main(int argc, char** argv) {
    using namespace tp;

    const CliArgs args = parseArgs(argc, argv);
    Config cfg;
    const fs::path project_dir = Config::projectDirectory();

    // 1. Detection engine: TFLite preferred, ONNX fallback.
    auto detector = createDetector(project_dir, cfg, args.model, args.threads);

    // 2. Capture source: camera or --file.
    std::unique_ptr<ThreadedCapture> capture;
    if (args.file.has_value()) {
        capture = std::make_unique<ThreadedCapture>(*args.file);
    } else {
        capture = std::make_unique<ThreadedCapture>(
            args.camera.value_or(cfg.camera_index),
            cfg.capture_width, cfg.capture_height, cfg.capture_fps,
            cfg.use_mjpeg);
    }

    CameraGeometry geometry(cfg.horizontal_fov);
    DistanceEstimator distance_estimator(
        cfg.horizontal_fov, cfg.player_height,
        (project_dir / "models" / "yunet_face_detection.onnx").string());

    capture->start();

    const bool show_gui = !args.no_gui;

    Detection last_person;
    long frame_index = 0;
    double fps = 0.0;
    auto fps_start_time = std::chrono::steady_clock::now();
    long fps_frame_count = 0;

    const int inference_every =
        (args.skip.value_or(cfg.frame_skip)) + 1;

    while (true) {
        cv::Mat frame = capture->read();

        if (frame.empty()) {
            if (capture->is_file() && capture->finished()) {
                std::cout << "Video ended" << std::endl;
                break;
            }
            // Camera/file not ready yet.
            if (show_gui && static_cast<char>(cv::waitKey(1)) == 'q') {
                break;
            }
            continue;
        }

        // Run inference only every (FRAME_SKIP + 1) frames. On skipped
        // frames the last known person box is reused.
        if (frame_index % inference_every == 0) {
            last_person = detector->detectPerson(
                frame, static_cast<float>(cfg.confidence_threshold));
        }
        ++frame_index;
        ++fps_frame_count;

        const auto now = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double>(now - fps_start_time).count();
        if (elapsed >= 1.0) {
            fps = fps_frame_count / elapsed;
            fps_frame_count = 0;
            fps_start_time = now;
        }

        if (!last_person.valid()) {
            if (show_gui) {
                cv::imshow("Tennis Player Detection", frame);
                if (static_cast<char>(cv::waitKey(1)) == 'q') {
                    break;
                }
            }
            continue;
        }

        const int x1 = last_person.x1;
        const int y1 = last_person.y1;
        const int x2 = last_person.x2;
        const int y2 = last_person.y2;

        if (show_gui) {
            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(0, 255, 0), 2);
        }

        const cv::Point player_center = geometry.getPlayerCenter(last_person);
        const cv::Point image_center = geometry.getImageCenter(frame);

        const double current_angle = geometry.calculateHorizontalAngle(
            player_center.x, image_center.x, frame.cols);

        const auto distance_result = distance_estimator.estimate(
            frame, last_person, frame.rows, frame.cols);

        // -------------------------------
        // PRINT REAL TIME ANGLE
        // -------------------------------
        if (!distance_result.first.has_value()) {
            std::printf(
                "Current rotation: %.2f degrees | Distance: N/A "
                "(player out of frame) | FPS: %.1f\n",
                current_angle, fps);
        } else {
            std::printf(
                "Current rotation: %.2f degrees | Distance: %.2f m | "
                "FPS: %.1f\n",
                current_angle, *distance_result.first, fps);
        }

        if (show_gui) {
            cv::circle(frame, player_center, 5, cv::Scalar(0, 0, 255), -1);
            cv::circle(frame, image_center, 5, cv::Scalar(255, 0, 0), -1);

            char angle_text[64];
            std::snprintf(angle_text, sizeof(angle_text),
                          "Angle: %.2f deg", current_angle);
            cv::putText(frame, angle_text, cv::Point(30, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(0, 255, 255), 2);

            cv::imshow("Tennis Player Detection", frame);
            if (static_cast<char>(cv::waitKey(1)) == 'q') {
                break;
            }
        }
    }

    capture->stop();

    if (show_gui) {
        cv::destroyAllWindows();
    }

    return 0;
}