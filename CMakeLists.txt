cmake_minimum_required(VERSION 3.24)

project(opencv_template_demo)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE=RELEASE)
set(BUILD_SHARED_LIBS=OFF)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# Check if OpenCV is already installed
find_package(OpenCV QUIET)

if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found. Fetching and building OpenCV...")

    set(BUILD_LIST=videoio,imgcodecs,highgui)
    set(WITH_EIGEN=ON)
    set(WITH_OPENMP=OFF)
    set(WITH_IPP=ON)
    set(WITH_NVCUVID=OFF)
    set(WITH_CUDA=OFF)
    set(WITH_CSTRIPES=ON)
    set(WITH_OPENCL=ON)
    set(BUILD_DOCS=OFF)
    set(BUILD_TESTS=OFF)
    set(BUILD_PERF_TESTS=OFF)
    set(BUILD_EXAMPLES=OFF)

    # Fetch OpenCV
    FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG 4.11.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(opencv)

    find_package(opencv REQUIRED)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(live_cam live_cam.cpp)
target_link_libraries(live_cam opencv_highgui opencv_imgcodecs opencv_core opencv_imgproc)

# link all opencv library modules
# target_link_libraries(live_cam ${OpenCV_LIBS})
