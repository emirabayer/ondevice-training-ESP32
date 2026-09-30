# "main" component: registers the bare-metal entry point plus the core
# engine's translation units directly from ../../src -- the engine is not
# copied/vendored into firmware/, this component just compiles the same
# .cpp files a second time for the xtensa-esp32s3 target. src/main.cpp
# (the host CLI demo, with its own `int main()`) is intentionally excluded;
# app_main.cpp is the bare-metal equivalent of that driver.

set(EDGE_CORE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

idf_component_register(
    SRCS
        "app_main.cpp"
        "${EDGE_CORE_ROOT}/src/arena.cpp"
        "${EDGE_CORE_ROOT}/src/layers.cpp"
        "${EDGE_CORE_ROOT}/src/optimizer.cpp"
    INCLUDE_DIRS
        "."
        "${EDGE_CORE_ROOT}/include"
    REQUIRES
        esp_timer
        esp_hw_support
        log
)

# The core engine is written to (and tested against) C++17; pin it
# explicitly rather than relying on whatever the active ESP-IDF version
# defaults its toolchain to.
target_compile_features(${COMPONENT_LIB} PUBLIC cxx_std_17)

# Same warning posture as the host build (CMakeLists.txt at the repo root).
target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra)
