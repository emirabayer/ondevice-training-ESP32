set(ENGINE "${CMAKE_CURRENT_LIST_DIR}/../..")
idf_component_register(
    SRCS "int8_train_main.cpp"
         "${ENGINE}/src/arena.cpp"
         "${ENGINE}/src/layers.cpp"
         "${ENGINE}/src/optimizer.cpp"
    INCLUDE_DIRS "." "${ENGINE}/include"
    REQUIRES esp_timer esp_system
)
