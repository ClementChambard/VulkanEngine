# required
cmake_minimum_required(VERSION 3.10)

# set some flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Debug")

# set the project name and version
project(VulkanTest VERSION 0.1)

# get source files
file(GLOB_RECURSE SRCS src/*.cpp src/*.c)
file(GLOB_RECURSE INCLUDES src/*.hpp src/*.h)

# add the executable
add_executable(VulkanTest ${SRCS} ${INCLUDES})

# add the build directory as an include directory (for the config header)
target_include_directories(VulkanTest PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "lib/stb"
                          )

# link libraries
target_link_libraries(VulkanTest PUBLIC
                      -lglfw
                      -lvulkan
                      -ldl
                      -lpthread
                      -lX11
                      -lXxf86vm
                      -lXrandr
                      -lXi
                    )

# Compile shaders
find_package(Vulkan REQUIRED COMPONENTS glslc)
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

file(GLOB SHADERS
     ${SHADER_SOURCE_DIR}/*.vert
     ${SHADER_SOURCE_DIR}/*.frag
     ${SHADER_SOURCE_DIR}/*.comp
     ${SHADER_SOURCE_DIR}/*.geom
     ${SHADER_SOURCE_DIR}/*.tesc
     ${SHADER_SOURCE_DIR}/*.tese
     ${SHADER_SOURCE_DIR}/*.mesh
     ${SHADER_SOURCE_DIR}/*.task
     ${SHADER_SOURCE_DIR}/*.rgen
     ${SHADER_SOURCE_DIR}/*.rchit
     ${SHADER_SOURCE_DIR}/*.rmiss
    )

add_custom_command(
  COMMAND
    ${CMAKE_COMMAND} -E make_directory ${SHADER_BINARY_DIR}
  OUTPUT ${SHADER_BINARY_DIR}
  COMMENT "Creating ${SHADER_BINARY_DIR}"
)

foreach(source IN LISTS SHADERS)
  get_filename_component(FILENAME ${source} NAME)
  add_custom_command(
    COMMAND
      ${glslc_executable}
      #      -MD -MF ${SHADER_BINARY_DIR}/${FILENAME}.d
      -o ${SHADER_BINARY_DIR}/${FILENAME}.spv
      ${source}
    OUTPUT ${SHADER_BINARY_DIR}/${FILENAME}.spv
    DEPENDS ${source} ${SHADER_BINARY_DIR}
    COMMENT "Compiling ${FILENAME}"
  )
  list(APPEND SPV_SHADERS ${SHADER_BINARY_DIR}/${FILENAME}.spv)
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_SHADERS})
