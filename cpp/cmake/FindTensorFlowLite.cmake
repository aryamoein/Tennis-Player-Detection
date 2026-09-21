# Finds the TensorFlow Lite C++ library (TensorFlowLite).
#
# Variables you can set:
#   TENSORFLOW_LITE_ROOT     - root of a TFLite install containing
#                              include/tensorflow/lite/model.h and
#                              lib/libtensorflow-lite.a (or libtflite.a).
#   TENSORFLOW_LITE_SRC_ROOT - (optional) TensorFlow source tree root, used
#                              when headers live in the source checkout and
#                              the library in a build dir (the standalone
#                              `lite/tools/make` build, for example).
#   TENSORFLOW_LITE_LIB_DIR  - (optional) directory holding the library when
#                              it is not directly under TENSORFLOW_LITE_ROOT.
#
# Outputs:
#   TENSORFLOW_LITE_FOUND
#   TENSORFLOW_LITE_INCLUDE_DIRS
#   TENSORFLOW_LITE_LIBRARIES

include(FindPackageHandleStandardArgs)

find_path(TENSORFLOW_LITE_INCLUDE_DIR
  NAMES tensorflow/lite/model.h
  HINTS ${TENSORFLOW_LITE_ROOT} ${TENSORFLOW_LITE_SRC_ROOT}
  PATH_SUFFIXES include
)

find_library(TENSORFLOW_LITE_LIBRARY
  NAMES tensorflow-lite tflite
  HINTS ${TENSORFLOW_LITE_ROOT} ${TENSORFLOW_LITE_LIB_DIR}
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(TensorFlowLite DEFAULT_MSG
  TENSORFLOW_LITE_INCLUDE_DIR
  TENSORFLOW_LITE_LIBRARY
)

mark_as_advanced(TENSORFLOW_LITE_INCLUDE_DIR TENSORFLOW_LITE_LIBRARY)

if(TENSORFLOW_LITE_FOUND)
  set(TENSORFLOW_LITE_INCLUDE_DIRS ${TENSORFLOW_LITE_INCLUDE_DIR})
  set(TENSORFLOW_LITE_LIBRARIES ${TENSORFLOW_LITE_LIBRARY})
endif()