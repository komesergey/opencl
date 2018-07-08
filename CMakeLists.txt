cmake_minimum_required(VERSION 3.10)
project(opencl)

add_executable(opencl main.cpp)
add_executable(devices devices.cpp)

message("CUDA: " $ENV{CUDA_PATH})

set(CMAKE_CXX_STANDARD 11)

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -framework OpenCL")
endif()

if(MSVC OR MSYS OR MINGW OR WIN32)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    if(DEFINED ENV{CUDA_PATH})
        message(STATUS "CUDA_PATH environment variable defined")
        include_directories("$ENV{CUDA_PATH}/include")
        target_link_libraries(opencl $ENV{CUDA_PATH}/lib/x64/OpenCL.lib)
        target_link_libraries(devices $ENV{CUDA_PATH}/lib/x64/OpenCL.lib)
    endif()
endif()