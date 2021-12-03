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

# Source group makes the files appear in same places in the logical structure
# of the project as where they are in the disk. This is for IDEs such as Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES src/cuda_thread_organization.cu)
set_property(TARGET cuda_thread_organization PROPERTY FOLDER "thread_organization")

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

# Source group makes the files appear in same places in the logical structure
# of the project as where they are in the disk. This is for IDEs such as Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES src/cuda_thread_1D_indexing.cu)
set_property(TARGET cuda_thread_1D_indexing PROPERTY FOLDER "thread_organization")

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

# Source group makes the files appear in same places in the logical structure
# of the project as where they are in the disk. This is for IDEs such as Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES src/cuda_thread_2D_indexing.cu)
set_property(TARGET cuda_thread_2D_indexing PROPERTY FOLDER "thread_organization")

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

# Source group makes the files appear in same places in the logical structure
# of the project as where they are in the disk. This is for IDEs such as Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES src/cuda_thread_3D_indexing.cu)
set_property(TARGET cuda_thread_3D_indexing PROPERTY FOLDER "thread_organization")
