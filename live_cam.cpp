#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <stdexcept>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "preprocess_frame.hpp"
#include "get_class_color.hpp"
#include "get_cameras.hpp"

int main(int argc, char **argv)
{
    if (argc < 1 + 2) {
        std::cerr << "Usage: " << argv[0]
                  << "<confidence-threshold> <non-max-suppression-threshold>\n";
        return 1;
    }

    char *err_ptr;

    // higher value will produce less detected objects due to higher detection passing rate
    float confidence_threshold = std::strtof(argv[1], &err_ptr);

    if (*err_ptr != '\0') {
        std::cerr << "invalid argument 1 (confidence threshold) value, not a number\n";
        return 1;
    }

    // non-max suppression threshold - higher value will produce more detected objects but can increase detection
    // redundancy
    float nms_threshold = std::strtof(argv[2], &err_ptr);

    if (*err_ptr != '\0') {
        std::cerr << "invalid argument 2 (NMS threshold) value, not a number\n";
        return 1;
    }

    printf("confidence threshold          : %.2f\n", confidence_threshold);
    printf("non-max suppression threshold : %.2f\n", nms_threshold);

    // ===============================================================================

    const std::string coco80_class[80] = {
      "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
      "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
      "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
      "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
      "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
      "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
      "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
      "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
      "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
      "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
      "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
      "teddy bear",     "hair drier", "toothbrush"
    };

    // ===============================================================================

    torch::jit::script::Module model = torch::jit::load("yolo11n.torchscript");

    torch::Device device(torch::kCPU);

    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    } else if (torch::mps::is_available()) {
        device = torch::Device(torch::kMPS);
    }

    model.to(device);
    model.eval();

    // ===============================================================================

    const int max_cam_to_check = 10; // adjust based on expected number of cameras

    std::vector<Camera> available_cameras = get_available_cameras(max_cam_to_check);

    std::cout << "found " << available_cameras.size() << " camera(s)\n";

    // ===============================================================================

    auto highest_resolution_camera = get_highest_resolution_cam(available_cameras);

    if (!highest_resolution_camera) {
        throw std::runtime_error("unable to find the highest resolution camera");
    }

    // ===============================================================================

    std::cout << "capturing from camera : " << highest_resolution_camera->camera_index << '\n';
    cv::VideoCapture cap(highest_resolution_camera->camera_index);
    int              current_camera_index = 0;
    std::cout << "capturing from camera : " << highest_resolution_camera->camera_index << '\n';

    if (!cap.isOpened()) {
        std::cerr << "not capturing\n";
        return -1;
    } else {
        std::cerr << "capturing\n";
        // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        // cap.set(cv::CAP_PROP_FPS, 45);
        // cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }

    cv::namedWindow("selected camera", cv::WINDOW_NORMAL);
    cv::Mat frame;

    double previous_time = static_cast<double>(cv::getTickCount());
    double fps = 0.0;

    std::vector<cv::Rect> origin_bounding_boxes;
    std::vector<float>    confidence_scores;
    std::vector<int>      class_ids;
    std::vector<int>      final_detection_indices;
    std::string           label;

    origin_bounding_boxes.reserve(8400 + 1);
    confidence_scores.reserve(8400 + 1);
    class_ids.reserve(8400 + 1);
    final_detection_indices.reserve(8400 + 1);
    label.reserve(50);

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

        // ------------------------------------------------------

        int origin_w = frame.cols;
        int origin_h = frame.rows;

        torch::Tensor input_tensor = preprocess_frame(frame);
        torch::Tensor output = model.forward({input_tensor.to(device)}).toTensor().to(torch::kCPU);
        output = output.squeeze(0).transpose(0, 1); // squeeze -> [84,  8400], transpose -> [8400, 84]    // 123911ns

        const float input_w = 640.f;
        const float input_h = 640.f;

        int num_detections = output.size(0); // 8400

        for (int i = 0; i < num_detections; ++i) {
            auto class_scores = output[i].slice(0, 4, 84);
            auto [max_score, max_index] = torch::max(class_scores, 0);

            float confidence = max_score.item<float>();

            if (confidence >= confidence_threshold) {
                int class_id = max_index.item<int>();

                // bounding box coordinates on the model's input image

                float bb_output_center_x = output[i][0].item<float>();
                float bb_output_center_y = output[i][1].item<float>();
                float bb_output_w = output[i][2].item<float>();
                float bb_output_h = output[i][3].item<float>();

                // converted bounding box coordinates on the original image

                float bb_origin_center_x = (bb_output_center_x / input_w) * origin_w;
                float bb_origin_center_y = (bb_output_center_y / input_h) * origin_h;

                float bb_origin_w = (bb_output_w / input_w) * origin_w;
                float bb_origin_h = (bb_output_h / input_h) * origin_h;

                float bb_origin_left = std::max(0.f, bb_origin_center_x - bb_origin_w / 2.f);
                float bb_origin_top = std::max(0.f, bb_origin_center_y - bb_origin_h / 2.f);

                cv::Rect origin_bounding_box(
                  static_cast<int>(bb_origin_left), static_cast<int>(bb_origin_top), static_cast<int>(bb_origin_w),
                  static_cast<int>(bb_origin_h)
                );

                origin_bounding_boxes.push_back(origin_bounding_box);
                confidence_scores.push_back(confidence);
                class_ids.push_back(class_id);
            }
        }

        std::vector<int> final_detection_indices;

        // remove duplicates bounding boxes
        cv::dnn::NMSBoxes(
          origin_bounding_boxes, confidence_scores, confidence_threshold, nms_threshold, final_detection_indices
        );

        for (int i: final_detection_indices) {
            cv::Rect origin_bounding_box = origin_bounding_boxes[i];
            float    confidence = confidence_scores[i];
            int      class_id = class_ids[i];

            cv::Scalar color = get_class_color(class_id);

            // draw bounding box with class-specific color

            cv::rectangle(frame, origin_bounding_box, color, 2);

            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << confidence;
            std::string score_str = ss.str();

            label.append(coco80_class[class_id]);
            label.append(": ");
            label.append(ss.str());

            int      baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

            // draw label background with the same class-specific color

            cv::rectangle(
              frame, cv::Point(origin_bounding_box.x, origin_bounding_box.y - text_size.height - 5),
              cv::Point(origin_bounding_box.x + text_size.width, origin_bounding_box.y), color, -1
            );

            // draw text in black for contrast

            cv::putText(
              frame, label, cv::Point(origin_bounding_box.x, origin_bounding_box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 0, 0), 2
            );

            label.resize(0);
        }

        origin_bounding_boxes.resize(0);
        confidence_scores.resize(0);
        class_ids.resize(0);
        final_detection_indices.resize(0);

        // ------------------------------------------------------

        std::ostringstream fps_stream;
        fps_stream << "FPS: " << std::fixed << std::setprecision(1) << fps;
        std::string fps_text = fps_stream.str();

        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

        cv::imshow("selected camera", frame);

        int key = cv::waitKey(1);

        if (key == 'q' || key == 'Q' || key == 27) { // 'q' or 'ESC' to exit
            std::cout << "close exit through key\n";
            break;
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
                std::cout << "Switched to camera index " << current_camera_index << "\n";
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
                std::cout << "Switched to camera index " << current_camera_index << "\n";
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}