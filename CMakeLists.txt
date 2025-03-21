cmake_minimum_required(VERSION 3.10)

project(opencv_template_demo)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

message(STATUS "CPU_BASELINE : ${CPU_BASELINE}")
message(STATUS "CPU_DISPATCH : ${CPU_DISPATCH}")

if(MSVC)
    add_compile_options(
        $<$<CONFIG:Release>:/MT>
        $<$<CONFIG:Debug>:/MTd>
    )
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(live_cam live_cam.cpp)
target_link_libraries(live_cam opencv_highgui opencv_imgcodecs opencv_core opencv_imgproc)

# link all opencv library modules
# target_link_libraries(live_cam ${OpenCV_LIBS})
