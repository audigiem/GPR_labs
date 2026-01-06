cmake_minimum_required(VERSION 3.0)

cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0072 NEW)

project(01-icp-base)

set (CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

execute_process(COMMAND ln -s ../shaders)

set(appName 01-icp-base)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLM REQUIRED)

include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${GLFW3_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIRS})
include_directories("${CMAKE_SOURCE_DIR}/gl3w")
include_directories("${CMAKE_SOURCE_DIR}/eigen")
include_directories("${CMAKE_SOURCE_DIR}/nanoflann")

link_directories(${OPENGL_LIBRARY_DIRS})
link_directories(${GLFW3_LIBRARY_DIRS})

link_libraries(dl)

set(imgui_files imgui/imconfig.h imgui/imgui.cpp imgui/imgui_demo.cpp imgui/imgui_draw.cpp imgui/imgui.h imgui/imgui_internal.h imgui/imgui_widgets.cpp imgui/imstb_rectpack.h imgui/imstb_textedit.h imgui/imstb_truetype.h imgui/imgui_impl_opengl3.h imgui/imgui_impl_opengl3.cpp imgui/imgui_impl_glfw.h imgui/imgui_impl_glfw.cpp imgui/imgui_tables.cpp) 

set(gl3w_files gl3w/gl3w.c gl3w/GL/gl3w.h)

#set(nanoflann_files nanoflann/nanoflann.hpp)

add_executable(${appName} ${imgui_files} ${gl3w_files} ${nanoflann_files} PointCloud.h PointCloud.cpp Camera.h Camera.cpp Scene.h Scene.cpp Shader.h Shader.cpp ShaderProgram.h ShaderProgram.cpp Application.h Application.cpp main.cpp NormalEstimator.h NormalEstimator.cpp NearestNeighbors.h NearestNeighbors.cpp IterativeClosestPoint.h IterativeClosestPoint.cpp SegmentCloud.h SegmentCloud.cpp)

target_link_libraries(${appName} ${OPENGL_LIBRARIES} glfw) 



