#------------------------
#--- CUDA hello world ---
#------------------------
add_executable(cuda_hello_world src/cuda_hello_world.cu)

set_property(TARGET cuda_hello_world PROPERTY CXX_STANDARD 17)

target_include_directories(cuda_hello_world
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cuda_hello_world CUDA::cudart)

# Source group makes the files appear in same places in the logical structure
# of the project as where they are in the disk. This is for IDEs such as Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES src/cuda_hello_world.cu)
set_property(TARGET cuda_hello_world PROPERTY FOLDER "hello_world")
