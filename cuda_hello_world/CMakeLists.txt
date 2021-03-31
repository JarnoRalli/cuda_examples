find_package(CUDAToolkit REQUIRED)

#--- CUDA hello world ---

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

