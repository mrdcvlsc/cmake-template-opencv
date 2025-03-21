cmake_minimum_required(VERSION 3.28)

project(sfml_opencv_libtorch_project)

message(STATUS "BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "CPU_BASELINE : ${CPU_BASELINE}")
message(STATUS "CPU_DISPATCH : ${CPU_DISPATCH}")

# ------------------------------------------------------------------
# INITIAL SETUPS
# ------------------------------------------------------------------

if(MSVC)
    if(BUILD_SHARED_LIBS)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
    endif()

    message(STATUS "CMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}")

    add_compile_options(
        $<$<CONFIG:Release>:/MT>
        $<$<CONFIG:Debug>:/MTd>
    )

    message(STATUS "CONFIG=${CONFIG}")
endif()

# ------------------------------------------------------------------
# SFML
# ------------------------------------------------------------------

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(SFML)

# ------------------------------------------------------------------
# OpenCV
# ------------------------------------------------------------------

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ------------------------------------------------------------------
# LIBTORCH
# ------------------------------------------------------------------

include(FetchContent)

set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_SILENT OFF)

# libtorch - get the libtorch shared binaries (THIS IS FOR LINUX - CPU)
message("--------------------------")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message("Operating System Type: LINUX")
  FetchContent_Declare(Torch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.0%2Bcpu.zip
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message("Operating System Type: APPLE")
  if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64")
      message(STATUS "x86_64 architecture detected")
      FetchContent_Declare(Torch
        URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.5.0.zip
      )
  elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm64")
      message(STATUS "arm64 architecture detected")
      FetchContent_Declare(Torch
        URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.0.zip
      )
  else()
      message(FATAL_ERROR "Other architecture detected")
  endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message("Operating System Type: WINDOWS")
  FetchContent_Declare(Torch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.5.0%2Bcpu.zip
  )
else()
  message("Operating System Type: Unknown")
  message(FATAL_ERROR "Cannot be supported")
endif()
message("--------------------------")

# libtorch - make the libtorch library available to the project
FetchContent_MakeAvailable(Torch)

# libtorch - add the lib paths of the downloaded library from FetchContent
list(APPEND CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}")

# libtorch - find the libtorch lib and require it for the project
find_package(Torch REQUIRED)

# libtorch - link the libtorch libraries
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ------------------------------------------------------------------
# live_cam_object_detection (OpenCV + libtorch)

add_executable(live_cam_object_detection live_cam_object_detection.cpp get_cameras.cpp event_keys.cpp preprocess_frame.cpp get_class_color.cpp logger.cpp)
target_link_libraries(live_cam_object_detection PRIVATE "${TORCH_LIBRARIES}")
# target_link_libraries(live_cam_object_detection PRIVATE opencv_highgui opencv_imgcodecs opencv_core opencv_imgproc)

# link all opencv library modules
target_link_libraries(live_cam_object_detection PRIVATE ${OpenCV_LIBS})
target_link_libraries(live_cam_object_detection PRIVATE SFML::Audio)

if (MSVC)
file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET live_cam_object_detection
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${TORCH_DLLS}
                      $<TARGET_FILE_DIR:live_cam_object_detection>)
endif (MSVC)

# ------------------------------------------------------------------
# live_cam_motion_detection (OpenCV only)

add_executable(live_cam_motion_detection live_cam_motion_detection.cpp get_cameras.cpp event_keys.cpp logger.cpp)
# target_link_libraries(live_cam_motion_detection PRIVATE "${TORCH_LIBRARIES}")
# target_link_libraries(live_cam_motion_detection PRIVATE opencv_highgui opencv_imgcodecs opencv_core opencv_imgproc)

# link all opencv library modules
target_link_libraries(live_cam_motion_detection PRIVATE ${OpenCV_LIBS})
target_link_libraries(live_cam_motion_detection PRIVATE SFML::Audio)

# if (MSVC)
# file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET live_cam_motion_detection
#                       POST_BUILD
#                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                       ${TORCH_DLLS}
#                       $<TARGET_FILE_DIR:live_cam_motion_detection>)
# endif (MSVC)


# ------------------------------------------------------------------
# draw_bounding_box (OpenCV + libtorch)

add_executable(draw_bounding_box draw_bounding_box.cpp preprocess_frame.cpp get_class_color.cpp event_keys.cpp logger.cpp)
target_link_libraries(draw_bounding_box PRIVATE "${TORCH_LIBRARIES}")
# target_link_libraries(draw_bounding_box PRIVATE opencv_highgui opencv_imgcodecs opencv_core opencv_imgproc)

# link all opencv library modules
target_link_libraries(draw_bounding_box PRIVATE ${OpenCV_LIBS})

if (MSVC)
file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET draw_bounding_box
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${TORCH_DLLS}
                      $<TARGET_FILE_DIR:draw_bounding_box>)
endif (MSVC)
