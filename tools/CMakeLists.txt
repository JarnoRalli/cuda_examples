# -----------------------------------
# Generate the library (i.e. target)
# -----------------------------------
list(   APPEND 
        src_files   
        src/timing.cpp
)

list(   APPEND 
        header_files
        include/tools/timing.hpp
        include/tools/error_handling.hpp
)

# Add a library
add_library(tools)

# Add directories to the target
# PRIVATE and PUBLIC items will populate the INCLUDE_DIRECTORIES property of target
# PUBLIC and INTERFACE items will populate the INTERFACE_INCLUDE_DIRECTORIES property of target
target_include_directories (tools
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Add sources to the library
# Header files are added only so that they appear in IDEs such as Visual Studio etc
# For INTERFACE libraries (header only) this is the only way to make the header
# files appear in the project in IDEs such as Visual Studio
target_sources(tools
    PRIVATE
        ${src_files}
        ${header_files}
)

# Source group makes the files appear in same places in the logical structure
# of the project as where they are in the disk. This is for IDEs such as Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${src_files} ${header_files})
set_property(TARGET tools PROPERTY FOLDER "tools")

set_property(TARGET tools PROPERTY CXX_STANDARD 17)

