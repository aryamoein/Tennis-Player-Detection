#include "config.hpp"

namespace tp {

fs::path Config::projectDirectory() {
    // cpp/src/config.cpp -> cpp/src -> cpp -> project root
    return fs::path(__FILE__).lexically_normal().parent_path().parent_path().parent_path();
}

fs::path Config::modelPath() const {
    return projectDirectory() / model_file;
}

fs::path Config::videoPath() const {
    return projectDirectory() / video_path;
}

}  // namespace tp