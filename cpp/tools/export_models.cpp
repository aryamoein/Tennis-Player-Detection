// Build detection models for the Raspberry Pi Zero 2 W.
//
// Mirrors tools/export_models.py: no ML toolchain required, downloads
// two ready-to-use int8 TFLite detection models from the Google Coral
// model zoo:
//
//   1. ssd_mobilenet_v2_coco_int8_300.tflite   (300x300 input)
//   2. efficientdet_lite0_coco_int8_320.tflite (320x320 input)
//
// Usage:
//     ./export_models
//
// Built with libcurl when available; otherwise falls back to the
// system `curl` binary.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#ifdef TP_HAVE_CURL
#include <curl/curl.h>
#endif

namespace {

namespace fs = std::filesystem;

const char* const kCorals[] = {
    "https://github.com/google-coral/test_data/raw/master/"
    "ssd_mobilenet_v2_coco_quant_postprocess.tflite",
    "https://github.com/google-coral/test_data/raw/master/"
    "efficientdet_lite0_320_ptq.tflite",
};

const char* const kTargets[] = {
    "ssd_mobilenet_v2_coco_int8_300.tflite",
    "efficientdet_lite0_coco_int8_320.tflite",
};

#ifdef TP_HAVE_CURL
std::size_t writeCallback(char* ptr, std::size_t size, std::size_t nmemb,
                          void* userdata) {
    FILE* file = static_cast<FILE*>(userdata);
    return std::fwrite(ptr, size, nmemb, file);
}

bool curlDownload(const std::string& url, const std::string& target,
                  bool allow_insecure) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }

    FILE* file = std::fopen(target.c_str(), "wb");
    if (!file) {
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, file);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
    if (allow_insecure) {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    }

    const CURLcode result = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    std::fclose(file);
    return result == CURLE_OK;
}
#else
// Fallback: rely on the system `curl` binary (with a TLS worst-case
// fallback, mirroring the Python certifi-unverified path).
bool curlDownload(const std::string& url, const std::string& target,
                  bool allow_insecure) {
    const std::string command =
        "curl -L --fail --connect-timeout 30 -o '" + target + "' '" + url + "'" +
        (allow_insecure ? " -k" : "");
    const int result = std::system(command.c_str());
    return result == 0;
}
#endif

void download(const std::string& url, const fs::path& target) {
    std::cout << target.filename().string() << std::endl;

    // TLS fallback (missing CA store / expired bundles): retry with
    // certificate verification disabled.
    bool ok = curlDownload(url, target.string(), false);
    if (!ok) {
        std::cout << "  certificate verification failed, retrying without it..."
                  << std::endl;
        std::remove(target.string().c_str());
        ok = curlDownload(url, target.string(), true);
    }

    if (!ok) {
        std::cout << "  ERROR: download failed: " << url << std::endl;
    } else {
        std::cout << "  -> " << target.string() << std::endl;
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    const fs::path project_directory =
        fs::path(__FILE__).lexically_normal().parent_path().parent_path().parent_path();
    const fs::path models_directory = project_directory / "models";

    fs::create_directories(models_directory);

    for (std::size_t i = 0; i < 2; ++i) {
        const fs::path target = models_directory / kTargets[i];
        if (fs::exists(target)) {
            std::cout << "  " << kTargets[i] << ": already present" << std::endl;
            continue;
        }
        download(kCorals[i], target);
    }

    std::cout << "\nDone. Benchmark the models on the Pi:\n"
              << "  cpp/build/benchmark\n\n"
              << "Pick the fastest and set it in src/config.py "
              << "(MODEL_FILE) or cpp/include/config.hpp, or pass "
              << "--model on the command line." << std::endl;

    return 0;
}