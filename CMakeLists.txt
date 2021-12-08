cmake_minimum_required(VERSION 3.12)

project(CUDA_examples LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75 CACHE STRING "CUDA architectures" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(tools)
add_subdirectory(cuda_hello_world)
add_subdirectory(cuda_thread_organization)
add_subdirectory(cuda_add)
add_subdirectory(cuda_device_properties)
add_subdirectory(cuda_warp)
add_subdirectory(cuda_resource_usage)
