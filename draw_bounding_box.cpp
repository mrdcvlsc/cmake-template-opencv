#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <iomanip>
#include <filesystem>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include "preprocess_frame.hpp"
#include "get_class_color.hpp"
#include "torch/types.h"

int main(int argc, char **argv)
{
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

    if (argc < 1 + 6) {
        std::cerr << "Usage: " << argv[0]
                  << "[confidence-threshold] [non-max-suppression-threshold] "
                     "[torch-script-jit-model] [model-input-dimension-size] "
                     "[input-image] [output-image]\n";
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

    std::string model_path = argv[3];
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "invalid argument 3, file not found (model: '" << model_path << "')\n";
        return 1;
    }

    // model_input_dim_size x model_input_dim_size
    float model_input_dim_size = std::strtof(argv[4], &err_ptr);

    if (*err_ptr != '\0') {
        std::cerr << "invalid argument 4 (model_input_dim_size) value, not a number\n";
        return 1;
    }

    std::string input_img_path = argv[5];
    if (!std::filesystem::exists(input_img_path)) {
        std::cerr << "invalid argument 5, file not found (input-img-path: '" << input_img_path << "')\n";
        return 1;
    }

    std::string output_img_path = argv[6];

    printf("confidence-threshold          : %.2f\n", confidence_threshold);
    printf("non-max-suppression-threshold : %.2f\n", nms_threshold);
    printf("torch-script-jit-model        : %s\n", model_path.c_str());
    printf("model-input-dimension-size    : %.2f\n", model_input_dim_size);
    printf("input-img-path                : %s\n", input_img_path.c_str());
    printf("output-img-path               : %s\n\n", output_img_path.c_str());

    printf("num_interop_threads : %d\n", torch::get_num_interop_threads());
    printf("num_threads         : %d\n", torch::get_num_threads());
    printf("cpu_capability      : %s\n\n", torch::get_cpu_capability().c_str());

    cv::Mat origin_img = cv::imread(input_img_path);
    if (origin_img.empty()) {
        std::cerr << "could not read image from " << input_img_path << "\n";
        return 1;
    }

    int origin_w = origin_img.cols;
    int origin_h = origin_img.rows;

    const float input_w = model_input_dim_size;
    const float input_h = model_input_dim_size;

    torch::Tensor input_tensor =
      preprocess_frame(origin_img, static_cast<int>(input_w), static_cast<int>(input_h), torch::kFloat32);

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
    }
    catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << '\n';
        return 1;
    }

    torch::Device device(torch::kCPU);

    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "using device                  : CUDA" << '\n';
    } else if (torch::mps::is_available()) {
        device = torch::Device(torch::kMPS);
        std::cout << "using device                  : MPS" << '\n';
    } else {
        std::cout << "using device                  : CPU" << '\n';
    }

    model.to(device);
    model.eval();

    torch::NoGradGuard no_grad;
    torch::Tensor      output;
    try {
        output = model.forward({input_tensor.to(device)}).toTensor().to(torch::kCPU);
    }
    catch (const c10::Error &e) {
        std::cerr << "error during inference: " << e.what() << '\n';
        return 1;
    }

    std::cout << "output tensor shape: " << output.sizes() << '\n'; // [1, 84, 8400]

    output = output.squeeze(0).transpose(0, 1); // squeeze -> [84,  8400], transpose -> [8400, 84]    // 123911ns

    std::cout << "transposed output shape: " << output.sizes() << '\n'; // [8400, 84]

    std::vector<cv::Rect> origin_bounding_boxes;
    std::vector<float>    confidence_scores;
    std::vector<int>      class_ids;

    int num_detections = output.size(0); // 8400

    std::cout << "processing " << num_detections << " potential detections" << '\n';

    auto start = std::chrono::system_clock::now();

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

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "duration = " << duration.count() << '\n';

    std::vector<int> final_detection_indices;

    // remove duplicates bounding boxes
    cv::dnn::NMSBoxes(
      origin_bounding_boxes, confidence_scores, confidence_threshold, nms_threshold, final_detection_indices
    );

    std::cout << "after NMS: " << final_detection_indices.size() << " detections" << '\n';

    std::string label;
    label.reserve(50);

    for (int i: final_detection_indices) {
        cv::Rect origin_bounding_box = origin_bounding_boxes[i];
        float    confidence = confidence_scores[i];
        int      class_id = class_ids[i];

        cv::Scalar color = get_class_color(class_id);

        // draw bounding box with class-specific color

        cv::rectangle(origin_img, origin_bounding_box, color, 2);

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
          origin_img, cv::Point(origin_bounding_box.x, origin_bounding_box.y - text_size.height - 5),
          cv::Point(origin_bounding_box.x + text_size.width, origin_bounding_box.y), color, -1
        );

        // draw text in black for contrast

        cv::putText(
          origin_img, label, cv::Point(origin_bounding_box.x, origin_bounding_box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
          cv::Scalar(0, 0, 0), 2
        );

        label.resize(0);
    }

    if (!cv::imwrite(output_img_path, origin_img)) {
        std::cerr << "failed to save output image to " << output_img_path << "\n";
        return 1;
    }

    std::cout << "output image saved to " << output_img_path << "\n";
    std::cout << "found " << final_detection_indices.size() << " objects" << '\n';

    return 0;
}