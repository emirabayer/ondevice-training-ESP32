cmake_minimum_required(VERSION 3.16)
# INT8 QAS + sparse-update gesture training for the YD-ESP32-S3.
# Freezes a pre-trained feature extractor, caches its INT8 features once, and
# trains ONLY the classifier head on-device using the custom 256KB-arena
# engine (INT8 matmul, QAS scale calibration, float-shadow Adam). Reports
# per-epoch time, arena high-water, and test accuracy for comparison with the
# naive FP32 baseline.
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(gesture_train_int8)
