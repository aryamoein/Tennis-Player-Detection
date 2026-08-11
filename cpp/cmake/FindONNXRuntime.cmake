# Finds the ONNX Runtime C++ API (ONNXRUNTIME).
#
# Variables you can set:
#   ONNXRUNTIME_ROOT  - root of an onnxruntime install containing
#                       include/onnxruntime_cxx_api.h and
#                       lib/{libonnxruntime.so, onnxruntime.lib, ...}
#
# Outputs:
#   ONNXRUNTIME_FOUND
#   ONNXRUNTIME_INCLUDE_DIRS
#   ONNXRUNTIME_LIBRARIES

include(FindPackageHandleStandardArgs)

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}
  PATH_SUFFIXES include
)

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(ONNXRUNTIME DEFAULT_MSG
  ONNXRUNTIME_INCLUDE_DIR
  ONNXRUNTIME_LIBRARY
)

mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)

if(ONNXRUNTIME_FOUND)
  set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
  set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY})
endif()