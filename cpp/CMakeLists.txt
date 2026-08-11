cmake_minimum_required(VERSION 3.16)
project(tennis_player_detection_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED
  COMPONENTS core imgproc imgcodecs videoio highgui objdetect)
find_package(Threads REQUIRED)

option(TP_ENABLE_ONNX "Enable the ONNX Runtime detection backend" ON)
option(TP_ENABLE_TFLITE "Enable the TensorFlow Lite detection backend" ON)

if(TP_ENABLE_ONNX)
  find_package(ONNXRuntime REQUIRED)
  find_package(OpenMP QUIET)
endif()

if(TP_ENABLE_TFLITE)
  find_package(TensorFlowLite QUIET)
endif()

find_package(CURL QUIET)

# ---------------------------------------------------------------------------
# tp_core  - OpenCV-dependent core (capture, geometry, distance, config)
# ---------------------------------------------------------------------------
add_library(tp_core STATIC
  src/config.cpp
  src/capture.cpp
  src/geometry.cpp
  src/distance_formula.cpp
  src/distance_estimator.cpp
)
target_include_directories(tp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tp_core PUBLIC ${OpenCV_LIBS} Threads::Threads)

# ---------------------------------------------------------------------------
# tp_detectors  - ONNX YOLO and TFLite detectors + engine factory
# ---------------------------------------------------------------------------
add_library(tp_detectors STATIC
  src/detector.cpp
  src/tflite_detector.cpp
  src/engine_factory.cpp
)
target_link_libraries(tp_detectors PUBLIC tp_core)

# tp_detectors needs ONNX headers only when the backend is enabled.
if(TP_ENABLE_ONNX AND ONNXRUNTIME_FOUND)
  target_include_directories(tp_detectors SYSTEM PUBLIC ${ONNXRUNTIME_INCLUDE_DIRS})
  target_link_libraries(tp_detectors PUBLIC ${ONNXRUNTIME_LIBRARIES})
  target_compile_definitions(tp_detectors PUBLIC TP_HAVE_ONNX=1)
  if(OpenMP_FOUND)
    target_link_libraries(tp_detectors PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if(TP_ENABLE_TFLITE AND TENSORFLOW_LITE_FOUND)
  target_include_directories(tp_detectors SYSTEM PUBLIC ${TENSORFLOW_LITE_INCLUDE_DIRS})
  target_link_libraries(tp_detectors PUBLIC ${TENSORFLOW_LITE_LIBRARIES})
  target_compile_definitions(tp_detectors PUBLIC TP_HAVE_TFLITE=1)
  # The tflite static archive needs POSIX symbol resolution services.
  target_link_libraries(tp_detectors PUBLIC ${CMAKE_DL_LIBS})
endif()

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(tennis_detector src/main.cpp)
target_link_libraries(tennis_detector PRIVATE tp_detectors tp_core)

add_executable(benchmark tools/benchmark.cpp)
target_link_libraries(benchmark PRIVATE tp_detectors tp_core)

add_executable(export_models tools/export_models.cpp)
if(CURL_FOUND)
  target_link_libraries(export_models PRIVATE CURL::libcurl)
  target_compile_definitions(export_models PRIVATE TP_HAVE_CURL=1)
endif()