cmake_minimum_required(VERSION 3.12)

project(CUDA_examples LANGUAGES CUDA CXX C)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(cuda_hello_world)
add_subdirectory(cuda_thread_organization)
add_subdirectory(cuda_add)

