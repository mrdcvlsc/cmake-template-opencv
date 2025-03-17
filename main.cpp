#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

struct Camera
{
    int camera_index;
    int channels;
    int width;
    int height;
};

int main()
{
    // ===============================================================================
    //                               check available camera
    // ===============================================================================

    std::vector<Camera> available_cameras;

    const int max_cam_to_check = 20; // adjust based on expected number of cameras
    for (int i = 0; i < max_cam_to_check; ++i) {
        cv::VideoCapture cap(i);
        if (cap.isOpened()) {
            cv::Mat frame;
            cap >> frame;

            if (frame.empty()) {
                std::cerr << "Frame captured was empty\n";
                return 1;
            }

            if (frame.size.dims() != 2) {
                std::cerr << "Detected camera frame dimension is not equal to 2\n";
                return 1;
            }

            available_cameras.push_back({
              /*camera_index : */ i,
              /*channels     : */ frame.channels(),
              /*width        : */ frame.size.p[1],
              /*height       : */ frame.size.p[0],
            });

            std::cout << "Camera[" << i << "] : ch(" << frame.channels()
                      << ")"
                         ", dim("
                      << frame.size.p[1] << 'x' << frame.size.p[0] << ")\n";

            cap.release();
        }
    }

    std::cout << "found " << available_cameras.size() << " camera(s)\n";

    // ===============================================================================
    //                           use highest resolution camera
    // ===============================================================================

    int    max_pixels = 0;
    Camera highest_resolution_camera;

    for (auto &camera: available_cameras) {
        int pixels = camera.height * camera.width;
        if (pixels > max_pixels) {
            max_pixels = pixels;
            highest_resolution_camera = camera;
        }
    }

    cv::VideoCapture cap(highest_resolution_camera.camera_index);

    if (!cap.isOpened()) {
        return -1;
    }

    cv::namedWindow("highest resolution camera", cv::WINDOW_NORMAL);
    cv::Mat frame;

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        cv::imshow("highest resolution camera", frame);

        if (cv::waitKey(1) >= 0) {
            break; // exit on key press
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}