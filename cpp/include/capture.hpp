#pragma once

#include <mutex>
#include <string>
#include <thread>

#include <opencv2/videoio.hpp>

namespace tp {

/**
 * Capture frames in a background thread and always expose the
 * most recent frame.
 *
 * Mirrors src/capture.py (ThreadedCapture):
 *
 *  - Camera capture never blocks inference.
 *  - Stale frames are dropped instead of queued (lowest latency).
 *  - A single reusable frame buffer avoids copies.
 */
class ThreadedCapture {
public:
    // Camera source (index).
    ThreadedCapture(int source,
                    int width = 0,
                    int height = 0,
                    double fps = 0.0,
                    bool use_mjpeg = true,
                    int buffersize = 1);

    // Video file source (path).
    explicit ThreadedCapture(const std::string& file_path, bool use_mjpeg = true);

    ~ThreadedCapture();

    ThreadedCapture(const ThreadedCapture&) = delete;
    ThreadedCapture& operator=(const ThreadedCapture&) = delete;

    /** Open the source and start the capture thread. */
    void start();

    /** Stop the capture thread and release the source. */
    void stop();

    /**
     * Most recent frame, or an empty Mat when no frame is ready
     * yet (or the video file has ended).
     */
    cv::Mat read();

    /** True when the source is a video file. */
    bool is_file() const;

    /** True when a video file has reached the end. */
    bool finished() const;

private:
    void loop();

    std::string file_path_;
    int source_index_ = -1;
    bool is_file_source_ = false;

    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;
    bool use_mjpeg_ = true;
    int buffersize_ = 1;

    cv::VideoCapture capture_;
    cv::Mat frame_;
    bool finished_ = false;
    bool running_ = false;
    double file_delay_ = 0.0;
    bool has_file_delay_ = false;

    mutable std::mutex mutex_;
    std::thread thread_;
};

}  // namespace tp