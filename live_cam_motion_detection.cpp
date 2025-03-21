#include <cstdlib>
#include <iostream>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/bgsegm.hpp>

#include <SFML/Audio.hpp>

#include <stdexcept>
#include <string>
#include <string_view>

#include "logger.hpp"
#include "get_cameras.hpp"
#include "event_keys.hpp"

int main(int argc, char **argv)
{
    // my env best args : ./build/bin/live_cam_motion_detection 100 200 1000 837
    if (argc < 1 + 4) {
        std::cerr
          << "Args: " << argv[0]
          << "[history:recommend=200] [sensitivity_threshold:recommend=16] "
             "[min_area:recommend=150] [flip: -1, 0, 1, others]\n\n\n"
             "history               - number of frames to learn background\n"
             "sensitivity_threshold - sensitivity (higher = less sensitive)\n"
             "flip                  - (-1 flip horizontal & vertical), (0 flip horizontal), (1 flip vertical)\n\n";
    }

    std::cout << "key controls:\n"
              << "q, Q, Esc  =  close live camera capture (close program).\n"
                 "c, C       =  toggle (on/off) actions for detections ('off' by default).\n"
                 "[          =  switch to previous camera.\n"
                 "]          =  switch to next camera.\n"
                 "s          =  toggle (on/off) detect shadow ('on' by default).\n"
              << "\n";

    char *err_ptr;

    int history = std::strtol(argv[1], &err_ptr, 10);
    if (*err_ptr != '\0') {
        std::cerr << "invalid argument 1 (history) value, not a number\n";
        return 1;
    }

    int sensitivity_threshold = std::strtol(argv[2], &err_ptr, 10);
    if (*err_ptr != '\0') {
        std::cerr << "invalid argument 2 (sensitivity_threshold) value, not a number\n";
        return 1;
    }

    int min_area = std::strtol(argv[3], &err_ptr, 10);
    if (*err_ptr != '\0') {
        std::cerr << "invalid argument 3 (min_area) value, not a number\n";
        return 1;
    }

    int flip_value = std::strtol(argv[4], &err_ptr, 10);
    if (*err_ptr != '\0') {
        std::cerr << "invalid argument 4 (flip_value) value, not a number\n";
        return 1;
    }

    const int max_cam_to_check = 10;

    std::vector<Camera> available_cameras = get_available_cameras(max_cam_to_check);
    std::cout << "found " << available_cameras.size() << " camera(s)\n";

    auto highest_resolution_camera = get_highest_resolution_cam(available_cameras);
    if (!highest_resolution_camera) {
        throw std::runtime_error("unable to find the highest resolution camera");
    }

    std::cout << "capturing from camera : " << highest_resolution_camera->camera_index << '\n';

    cv::VideoCapture cap(highest_resolution_camera->camera_index);
    int              current_camera_index = 0;

    if (!cap.isOpened()) {
        std::cerr << "not capturing\n";
        return -1;
    } else {
        std::cerr << "capturing\n";
    }

    bool is_detect_shadow_on = true;

    cv::Ptr<cv::BackgroundSubtractorMOG2> bg_subtractor =
      cv::createBackgroundSubtractorMOG2(history, sensitivity_threshold, is_detect_shadow_on);

    cv::namedWindow("selected camera", cv::WINDOW_NORMAL);
    cv::Mat frame, fg_mask; // frame & foreground mask

    double previous_time = static_cast<double>(cv::getTickCount());
    double fps = 0.0;

    bool is_enable_action = false;

    sf::SoundBuffer sound_buffer("alert.wav");
    sf::Sound       alert_sound(sound_buffer);

    alert_sound.play();

    while (true) {
        double current_time = static_cast<double>(cv::getTickCount());
        double time_elapsed = (current_time - previous_time) / cv::getTickFrequency();

        if (time_elapsed > 0) {
            fps = 1.0 / time_elapsed;
        }
        previous_time = current_time;

        cap >> frame;
        if (frame.empty()) {
            throw std::runtime_error("detected an empty frame");
        }

        bg_subtractor->apply(frame, fg_mask);

        // optional: apply morphological operations to clean up the mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_OPEN, kernel);  // remove small noise
        cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_CLOSE, kernel); // close small gaps

        // Find contours in the foreground mask
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fg_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        bool motion_detected = false;

        // draw bounding boxes around significant contours
        for (const auto &contour: contours) {
            if (cv::contourArea(contour) > min_area) {
                cv::Rect bounding_box = cv::boundingRect(contour);
                cv::rectangle(frame, bounding_box, cv::Scalar(0, 0, 255), 2);
                motion_detected = true;
            }
        }

        if (motion_detected) {
            if (alert_sound.getStatus() != sf::Sound::Status::Playing) {
                alert_sound.play();
            }

            debug_log("motion detected");

            std::string motion_frame_img = get_timestamp() + ".jpg";

            if (!cv::imwrite(motion_frame_img, frame)) {
                std::cerr << "failed to save output image to " << motion_frame_img << "\n";
                return 1;
            }
        }

        std::ostringstream fps_stream;
        fps_stream << "FPS: " << std::fixed << std::setprecision(1) << fps;

        if (-1 >= flip_value && flip_value <= 1) {
            cv::Mat flipped_frame;
            cv::flip(frame, flipped_frame, flip_value);

            cv::putText(
              flipped_frame, fps_stream.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0),
              2
            );

            cv::imshow("selected camera", flipped_frame);
        } else {
            cv::putText(
              frame, fps_stream.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2
            );

            cv::imshow("selected camera", frame);
        }

        int key = cv::waitKey(1);

        if (key == 's') {
            bg_subtractor->clear();
            bg_subtractor = cv::createBackgroundSubtractorMOG2(history, sensitivity_threshold, is_detect_shadow_on);

            debug_log(
              "toggle shadow detection : " + std::string((is_detect_shadow_on) ? "true" : "false") + " -> " +
              ((is_detect_shadow_on) ? "false" : "true")
            );

            is_detect_shadow_on = !is_detect_shadow_on;
        }

        bool is_stop = check_events(key, cap, available_cameras, current_camera_index, is_enable_action);

        if (is_stop) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}