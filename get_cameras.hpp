#ifndef MRDCVLSC_OPENCV_CMAKE_TEMPLATE_GET_CAMERAS_HPP
#define MRDCVLSC_OPENCV_CMAKE_TEMPLATE_GET_CAMERAS_HPP

#include <vector>
#include <memory>

struct Camera
{
    int camera_index;
    int channels;
    int width;
    int height;
};

std::vector<Camera> get_available_cameras(int max_cam_to_check);
std::unique_ptr<Camera> get_highest_resolution_cam(const std::vector<Camera>& cameras);

#endif