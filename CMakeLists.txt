cmake_minimum_required(VERSION 3.14)
project(moecut C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# moecut never runs inference: CPU-only ggml, consumed for the gguf API and type traits
set(BUILD_SHARED_LIBS OFF)
set(GGML_CUDA   OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_METAL  OFF CACHE BOOL "" FORCE)
set(GGML_BLAS   OFF CACHE BOOL "" FORCE)
add_subdirectory(vendor/ggml EXCLUDE_FROM_ALL)

add_executable(moecut
    src/main.c
    src/common.c
    src/info.c
    src/profile.c
    src/prune.c
)
target_link_libraries(moecut PRIVATE ggml m)
target_compile_options(moecut PRIVATE -Wall -Wextra)

# ASan is non-negotiable for pointer arithmetic over multi-GB mmapped files
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(moecut PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(moecut PRIVATE -fsanitize=address,undefined)
endif()
