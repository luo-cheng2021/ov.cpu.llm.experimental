# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required(VERSION 3.23)

set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "ov-cpu-llm-experimental")

project($TARGET_NAME)

find_package(OpenVINO REQUIRED)

set(SRC ov_extension.cpp mha.cpp fc.cpp)

add_library(${TARGET_NAME} MODULE ${SRC})

# additional requirement TBB & avx512
#find_package(TBB QUIET COMPONENTS tbb)
#target_link_libraries(${TARGET_NAME} PUBLIC ${TBB_IMPORTED_TARGETS})

target_compile_options(${TARGET_NAME} PRIVATE "-mavx2")
target_compile_options(${TARGET_NAME} PRIVATE "-mfma")
target_compile_options(${TARGET_NAME} PRIVATE "-mf16c")
target_compile_options(${TARGET_NAME} PRIVATE "-mavxvnni")
target_compile_options(${TARGET_NAME} PRIVATE "-mno-avx256-split-unaligned-load")
target_compile_options(${TARGET_NAME} PRIVATE "-mno-avx256-split-unaligned-store")

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

