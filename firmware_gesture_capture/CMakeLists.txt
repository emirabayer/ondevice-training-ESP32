cmake_minimum_required(VERSION 3.16)
# Gesture-capture firmware for the YD-ESP32-S3 (N16R8).
# Streams 4 capacitive-touch channels (GPIO 7,8,9,12) over serial so the host
# can record labeled gesture data. See tools/collect_gestures.py.
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(gesture_capture)
