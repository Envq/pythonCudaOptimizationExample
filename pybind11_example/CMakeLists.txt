cmake_minimum_required(VERSION 3.4)
project(cuda_accelerations LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


list(APPEND CUDA_NVCC_FLAGS "-arch=sm_61")
list(APPEND CUDA_NVCC_FLAGS "-w")


# find_package(pybind11 REQUIRED)
find_package(CUDA  REQUIRED)

set(INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_transpose.cu
)

add_subdirectory(pybind11)
pybind11_add_module(cuda_accelerations ${SOURCES})
target_include_directories(cuda_accelerations PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(cuda_accelerations PRIVATE ${INCLUDES})