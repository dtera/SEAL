# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.13)

set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(SEAL_GPU VERSION 4.1.2 LANGUAGES CXX CUDA)

file(GLOB_RECURSE CURRENT_HEADERS *.h *.hpp *.cuh)
file(GLOB_RECURSE CURRENT_SOURCES *.cpp *.cu *.c)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")

add_library(seal_gpu SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
if (TARGET SEAL::seal)
    target_link_libraries(seal_gpu PRIVATE SEAL::seal)
elseif (TARGET SEAL::seal_shared)
    target_link_libraries(seal_gpu PRIVATE SEAL::seal_shared)
endif ()
set_target_properties(seal_gpu PROPERTIES
        CXX_STANDARD ${CMAKE_CUDA_STANDARD}
        CXX_EXTENSIONS OFF
        CXX_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON)

set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU>")
set(nvcc_cxx "$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>")
target_compile_options(seal_gpu INTERFACE
        "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-fPIC>>"
        "$<${nvcc_cxx}:$<BUILD_INTERFACE:--compiler-options=-fPIC>>"
)

install(TARGETS seal_gpu
        EXPORT sealTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

file(GLOB MAIN_DIR_HEADERS *.h *.hpp *.cuh)
file(GLOB UTILS_HEADERS utils/*.h *.hpp *.cuh)

install(FILES ${MAIN_DIR_HEADERS} DESTINATION include/seal_gpu)
install(FILES ${UTILS_HEADERS} DESTINATION include/seal_gpu/utils)