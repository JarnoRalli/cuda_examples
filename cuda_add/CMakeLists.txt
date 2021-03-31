find_package(CUDAToolkit REQUIRED)

#--- Add vectors example ---
add_executable(add src/add.cpp)

set_property(TARGET add PROPERTY CXX_STANDARD 17)

target_include_directories(add
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

#--- Add vectors example, CUDA version ---
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75)
endif()

add_executable(cuda_add src/cuda_add.cu)

set_property(TARGET cuda_add PROPERTY CXX_STANDARD 17)

target_include_directories(cuda_add
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cuda_add CUDA::cudart)

