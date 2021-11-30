find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75)
endif()

#---------------------------------
#--- Query CUDA device properties
#---------------------------------
add_executable(cuda_device_query src/cuda_device_query.cu)

set_property(TARGET cuda_device_query PROPERTY CXX_STANDARD 17)

target_include_directories(cuda_device_query
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cuda_device_query CUDA::cudart)

