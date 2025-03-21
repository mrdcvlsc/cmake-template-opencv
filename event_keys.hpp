#ifndef MRDCVLSC_OPENCV_CMAKE_TEMPLATE_EVENTS_HPP
#define MRDCVLSC_OPENCV_CMAKE_TEMPLATE_EVENTS_HPP

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "types.hpp"

// returns true if quit events are sent
bool check_events(
  int key, cv::VideoCapture &cap, const std::vector<Camera> &available_cameras, int &current_camera_index,
  bool &is_enable_action
);

#endif