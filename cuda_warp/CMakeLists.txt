find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75)
endif()

#---------------------------------
#--- Analyze CUDA warp divergence
#---------------------------------
add_executable(cuda_warp_divergence src/cuda_warp_divergence.cu)

set_property(TARGET cuda_warp_divergence PROPERTY CXX_STANDARD 17)

# -G switch disables device optionzations. This is done in order to demonstrate warp divergence
target_compile_options(cuda_warp_divergence PRIVATE "-G")

target_include_directories(cuda_warp_divergence
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cuda_warp_divergence CUDA::cudart)

