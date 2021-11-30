find_package(CUDAToolkit REQUIRED)

#--------------------------------
#--- CUDA thread organization ---
#--------------------------------

add_executable(cuda_thread_organization src/cuda_thread_organization.cu)

target_include_directories(cuda_thread_organization
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cuda_thread_organization CUDA::cudart tools)

#-------------------------------
#--- CUDA thread 1D indexing ---
#-------------------------------

add_executable(cuda_thread_1D_indexing src/cuda_thread_1D_indexing.cu)

target_include_directories(cuda_thread_1D_indexing
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cuda_thread_1D_indexing CUDA::cudart tools)

#-------------------------------
#--- CUDA thread 2D indexing ---
#-------------------------------

add_executable(cuda_thread_2D_indexing src/cuda_thread_2D_indexing.cu)

#set_property(TARGET cuda_thread_2D_indexing PROPERTY CXX_STANDARD 17)

target_include_directories(cuda_thread_2D_indexing
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cuda_thread_2D_indexing CUDA::cudart tools)

#-------------------------------
#--- CUDA thread 3D indexing ---
#-------------------------------

add_executable(cuda_thread_3D_indexing src/cuda_thread_3D_indexing.cu)

#set_property(TARGET cuda_thread_3D_indexing PROPERTY CXX_STANDARD 17)

target_include_directories(cuda_thread_3D_indexing
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cuda_thread_3D_indexing CUDA::cudart tools)

