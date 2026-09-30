cmake_minimum_required(VERSION 3.16)
# Naive FP32 gesture-training baseline for the YD-ESP32-S3.
# Trains the whole 64->32->5 MLP from scratch in float32, on-device, and
# reports per-epoch time + working-memory footprint + test accuracy. This is
# the baseline the INT8 QAS+sparse version is compared against.
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(gesture_train_fp32)
