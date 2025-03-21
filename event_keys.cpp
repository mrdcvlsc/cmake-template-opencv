#include <iostream>

#include "logger.hpp"
#include "event_keys.hpp"
/**
key press events:

q, Q, Esc  =  close live camera capture (close program).
c, C       =  enable actions for detections.
[          =  switch to previous camera.
]          =  switch to next camera.
*/
bool check_events(
  int key, cv::VideoCapture &cap, const std::vector<Camera> &available_cameras, int &current_camera_index,
  bool &is_enable_action
)
{
    if (key == 'q' || key == 'Q' || key == 27) { // 'q' or 'ESC' to exit
        debug_log("close exit through key");
        return true;
    }

    if (key == 'c' || key == 'C') {
        debug_log(
          "is_enable_action : " + std::string((is_enable_action) ? "true" : "false") + " -> " +
          ((is_enable_action) ? "false" : "true")
        );
        is_enable_action = !is_enable_action;
    }

    if (key == '[') { // previous camera
        if (current_camera_index > 0) {
            current_camera_index--;
        } else {
            current_camera_index = available_cameras.size() - 1; // Loop back to last camera
        }

        cap.release();
        cap.open(available_cameras[current_camera_index].camera_index);

        if (!cap.isOpened()) {
            std::cerr << "Failed to open camera index " << current_camera_index << "\n";
        } else {
            debug_log("Switched to camera index " + std::to_string(current_camera_index));
        }
    }

    if (key == ']') { // next camera
        if (current_camera_index < available_cameras.size() - 1) {
            current_camera_index++;
        } else {
            current_camera_index = 0;
        }

        cap.release();
        cap.open(available_cameras[current_camera_index].camera_index);

        if (!cap.isOpened()) {
            std::cerr << "Failed to open camera index " << current_camera_index << "\n";
        } else {
            debug_log("Switched to camera index " + std::to_string(current_camera_index));
        }
    }

    return false;
}