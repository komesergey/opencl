cmake_minimum_required(VERSION 3.10)
project(opencl)

set(CMAKE_CXX_STANDARD 11)

if(MSVC OR MSYS OR MINGW)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -framework OpenCL")
endif()


add_executable(opencl main.cpp)
add_executable(devices devices.cpp)