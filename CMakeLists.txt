cmake_minimum_required(VERSION 3.10)

project(opencv_template_demo)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp)
target_link_libraries(main opencv_highgui opencv_imgcodecs opencv_core opencv_imgproc)

# link all opencv library modules
# target_link_libraries(main ${OpenCV_LIBS})
