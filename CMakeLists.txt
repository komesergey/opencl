cmake_minimum_required(VERSION 3.10)
project(opencl)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -framework OpenCL")

add_executable(opencl main.cpp)
add_executable(devices devices.cpp)