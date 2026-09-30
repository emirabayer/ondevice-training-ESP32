idf_component_register(
    SRCS "gesture_capture_main.c"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_touch_sens esp_timer
)
