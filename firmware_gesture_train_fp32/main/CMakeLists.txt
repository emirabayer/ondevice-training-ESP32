idf_component_register(
    SRCS "fp32_train_main.cpp"
    INCLUDE_DIRS "." "${CMAKE_CURRENT_LIST_DIR}/../../include"
    REQUIRES esp_timer
)
