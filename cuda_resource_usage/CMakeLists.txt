#------------------------------------------------------------------------------------
#--- Add vectors example using grid stride loop, CUDA version with managed memory ---
#------------------------------------------------------------------------------------
add_executable(resource_usage_add_managed_memory src/cuda_add_managed_memory.cu)

target_include_directories(resource_usage_add_managed_memory
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(resource_usage_add_managed_memory CUDA::cudart tools)

# Source group makes the files appear in same places in the logical structure
# of the project as where they are in the disk. This is for IDEs such as Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES src/cuda_add_managed_memory.cu)
set_property(TARGET resource_usage_add_managed_memory PROPERTY FOLDER "resource_usage")

# --ptxas-options=-v outputs the number of registers usage per thread and shared memory usage per block
# This information is util when trying to determine occupancy of a CUDA device
target_compile_options(resource_usage_add_managed_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
