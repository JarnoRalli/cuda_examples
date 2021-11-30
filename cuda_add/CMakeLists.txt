find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75)
endif()

#---------------------------
#--- Add vectors example ---
#---------------------------
add_executable(add src/add.cpp)

set_property(TARGET add PROPERTY CXX_STANDARD 17)

target_include_directories(add
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

#---------------------------------------------------------------------------------------------
#--- Add vectors example using grid stride loop, CUDA version with manual memory transfers ---
#---------------------------------------------------------------------------------------------
add_executable(cuda_add_manual_memory src/cuda_add_manual_memory.cu)

#set_property(TARGET cuda_add_manual_memory PROPERTY CXX_STANDARD 17)

target_include_directories(cuda_add_manual_memory
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cuda_add_manual_memory CUDA::cudart tools)

#------------------------------------------------------------------------------------------------
#--- Add vectors example using a monolithic kernel, CUDA version with manual memory transfers ---
#------------------------------------------------------------------------------------------------
add_executable(cuda_add_manual_memory_monolithic src/cuda_add_manual_memory_monolithic.cu)

#set_property(TARGET cuda_monolithic_add_manual_memory PROPERTY CXX_STANDARD 17)

target_include_directories(cuda_add_manual_memory_monolithic
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cuda_add_manual_memory_monolithic CUDA::cudart tools)

#-----------------------------------------------------------------------------------
#--- Add vectors example using grid stride loop, CUDA version with managed memory---
#-----------------------------------------------------------------------------------
add_executable(cuda_add_managed_memory src/cuda_add_managed_memory.cu)

#set_property(TARGET cuda_add_managed_memory PROPERTY CXX_STANDARD 17)

target_include_directories(cuda_add_managed_memory
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cuda_add_managed_memory CUDA::cudart tools)

