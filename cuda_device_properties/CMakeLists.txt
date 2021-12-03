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

# Source group makes the files appear in same places in the logical structure
# of the project as where they are in the disk. This is for IDEs such as Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES src/cuda_device_query.cu)
set_property(TARGET cuda_device_query PROPERTY FOLDER "device_properties")
