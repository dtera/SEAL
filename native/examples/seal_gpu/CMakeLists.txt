# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.13)

set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86)
project(SEALGPUExamples VERSION 4.1.2 LANGUAGES CXX CUDA)

# If not called from root CMakeLists.txt
if(NOT DEFINED SEAL_BUILD_EXAMPLES)
    set(SEAL_BUILD_EXAMPLES ON)

    # Import Microsoft SEAL
    find_package(SEAL 4.1.2 EXACT REQUIRED)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()

if(SEAL_BUILD_EXAMPLES)
    file(GLOB_RECURSE EXAMPLES_SRCS *.c*)

    set(EXE_TARGETS)
    # Building examples
    foreach (examples_file ${EXAMPLES_SRCS})
        if (NOT ${examples_file} MATCHES ".*\.cu" OR SEAL_ENABLE_GPU)
            string(REGEX REPLACE "(.cc)|(.cpp)|(.cu)" "" examples_name ${examples_file})
            string(REGEX REPLACE "(.*)?/" "" examples_name ${examples_name})
            add_executable(${PROJECT_NAME}_${examples_name} ${examples_file})
            if (${examples_file} MATCHES ".*\.cu")
                set(SEAL_GPU_LIB seal_gpu)
            else ()
                set(SEAL_GPU_LIB)
            endif ()
            if(TARGET SEAL::seal)
                target_link_libraries(${PROJECT_NAME}_${examples_name} PRIVATE SEAL::seal ${SEAL_GPU_LIB})
            elseif(TARGET SEAL::seal_shared)
                target_link_libraries(${PROJECT_NAME}_${examples_name} PRIVATE SEAL::seal_shared ${SEAL_GPU_LIB})
            else()
                message(FATAL_ERROR "Cannot find target SEAL::seal or SEAL::seal_shared")
            endif()
            list(APPEND EXE_TARGETS ${PROJECT_NAME}_${examples_name})
        endif ()
    endforeach ()
    foreach (target ${EXE_TARGETS})
        message("target: ${target}")
    endforeach ()
    set_target_properties(${EXE_TARGETS} PROPERTIES
            CXX_STANDARD 17
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
    )
endif()
