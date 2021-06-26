cmake_minimum_required(VERSION 2.6)

project(slim)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/yc-mc/Downloads/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/include)
link_directories(/home/yc-mc/Downloads/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/lib)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(slim ${PROJECT_SOURCE_DIR}/slim.cpp)
target_link_libraries(slim nvinfer)
target_link_libraries(slim cudart)
target_link_libraries(slim ${OpenCV_LIBS})
add_definitions(-O2 -pthread)

