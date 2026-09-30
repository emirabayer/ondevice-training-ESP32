cmake_minimum_required(VERSION 3.16)
project(EdgeQuantTrain LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

add_compile_options(-Wall -Wextra -Werror)

option(EDGE_ENABLE_SANITIZERS "Enable AddressSanitizer/UndefinedBehaviorSanitizer in Debug builds" ON)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND EDGE_ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# Core engine: arena allocator, tensor/kernels, layers, QAS, optimizer.
# Everything here must run within the fixed 256KB arena -- no malloc/new/vector/string
# in these translation units outside of the arena's own backing buffer.
add_library(edge_core STATIC
  src/arena.cpp
  src/layers.cpp
  src/optimizer.cpp
)
target_include_directories(edge_core PUBLIC include)

add_executable(train src/main.cpp)
target_link_libraries(train PRIVATE edge_core)

enable_testing()
add_subdirectory(tests)
