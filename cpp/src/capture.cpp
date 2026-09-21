#include "capture.hpp"

#include <chrono>
#include <stdexcept>

namespace tp {

ThreadedCapture::ThreadedCapture(int source,
                                 int width, int height, double fps,
                                 bool use_mjpeg, int buffersize)
    : source_index_(source),
      is_file_source_(false),
      width_(width),
      height_(height),
      fps_(fps),
      use_mjpeg_(use_mjpeg),
      buffersize_(buffersize) {}

ThreadedCapture::ThreadedCapture(const std::string& file_path, bool use_mjpeg)
    : file_path_(file_path),
      is_file_source_(true),
      use_mjpeg_(use_mjpeg),
      buffersize_(1) {}

ThreadedCapture::~ThreadedCapture() {
    stop();
}

void ThreadedCapture::start() {
    const bool opened = is_file_source_ ? capture_.open(file_path_)
                                        : capture_.open(source_index_);
    if (!opened) {
        throw std::runtime_error(
            "Could not open video source: " +
            (is_file_source_ ? file_path_ : std::to_string(source_index_)));
    }

    if (!is_file_source_) {
        if (use_mjpeg_) {
            capture_.set(cv::CAP_PROP_FOURCC,
                         cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        }
        if (width_) {
            capture_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
        }
        if (height_) {
            capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
        }
        if (fps_ > 0.0) {
            capture_.set(cv::CAP_PROP_FPS, fps_);
        }
        capture_.set(cv::CAP_PROP_BUFFERSIZE, buffersize_);
        has_file_delay_ = false;
    } else {
        // Throttle video playback to its native speed so the whole
        // file does not finish instantly.
        const double playback_fps = capture_.get(cv::CAP_PROP_FPS);
        if (playback_fps > 0.0) {
            file_delay_ = 1.0 / playback_fps;
            has_file_delay_ = true;
        } else {
            has_file_delay_ = false;
        }
    }

    running_ = true;
    thread_ = std::thread(&ThreadedCapture::loop, this);
}

void ThreadedCapture::loop() {
    while (running_) {
        cv::Mat frame;
        const bool success = capture_.read(frame);
        if (!success) {
            // A video has ended; stop the thread.
            // A camera had a hiccup; keep trying.
            if (is_file_source_) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    finished_ = true;
                }
                break;
            }
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            frame_ = frame;
        }

        if (has_file_delay_) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(file_delay_));
        }
    }
}

cv::Mat ThreadedCapture::read() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (finished_) {
        return cv::Mat();
    }
    return frame_.clone();
}

bool ThreadedCapture::is_file() const {
    return is_file_source_;
}

bool ThreadedCapture::finished() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return finished_;
}

void ThreadedCapture::stop() {
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
    if (capture_.isOpened()) {
        capture_.release();
    }
}

}  // namespace tp