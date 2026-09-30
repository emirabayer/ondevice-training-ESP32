cmake_minimum_required(VERSION 3.16)

# Root ESP-IDF project file. This does NOT reuse the host CMakeLists.txt at
# the repository root: ESP-IDF projects are built exclusively through
# idf.py's own CMake driver (component-based, cross-compiling for the
# target chip), which is incompatible with a plain `add_executable`
# top-level build. The actual engine source is not duplicated here -- see
# main/CMakeLists.txt, which points straight at ../../include and ../../src.
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(edge_quant_train_firmware)
