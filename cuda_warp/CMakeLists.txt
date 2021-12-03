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

# Source group makes the files appear in same places in the logical structure
# of the project as where they are in the disk. This is for IDEs such as Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES src/cuda_warp_divergence.cu)
set_property(TARGET cuda_warp_divergence PROPERTY FOLDER "warp")
