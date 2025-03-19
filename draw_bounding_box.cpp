#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <iomanip>
#include <cmath>

torch::Tensor preprocessFrame(cv::Mat &frame)
{
    cv::Mat img;
    cv::resize(frame, img, cv::Size(640, 640));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    torch::Tensor tensor_image = torch::from_blob(img.data, {1, 640, 640, 3}, torch::kFloat32);
    tensor_image = tensor_image.permute({0, 3, 1, 2});
    return tensor_image.clone();
}

// Function to generate a unique color for each class ID
cv::Scalar getColorForClass(int class_id)
{
    // Use HSV color space where Hue varies by class ID
    float   hue = std::fmod(class_id * 137.0f, 360.0f); // 137 is a prime number for good distribution
    cv::Mat hsv(1, 1, CV_32FC3);
    hsv.at<cv::Vec3f>(0, 0) = cv::Vec3f(hue, 1.0f, 1.0f); // Full saturation and value
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3f color = bgr.at<cv::Vec3f>(0, 0);
    return cv::Scalar(color[0] * 255, color[1] * 255, color[2] * 255);
}

int main(int argc, char **argv)
{
    const std::string class_names[80] = {
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

    if (argc < 1 + 4) {
        std::cerr << "Usage: " << argv[0]
                  << "<confidence-threshold> <non-max-suppression-threshold> <input_image> <output_image>\n";
        return 1;
    }

    char *err_ptr;

    // Higher value will produce less detected objects due to higher detection passing rate
    float confidence_threshold = std::strtof(argv[1], &err_ptr);
    
    if (*err_ptr == '\0') {
        printf("%.2f\n", confidence_threshold);
    } else {
        puts("`arg1' is not a number!");
        return 1;
    }

    // Higher value will produce more detected objects but can increase detection redundancy
    float non_max_suppression_threshold = std::strtof(argv[2], &err_ptr);
    
    if (*err_ptr == '\0') {
        printf("%.2f\n", non_max_suppression_threshold);
    } else {
        puts("`arg2' is not a number!");
        return 1;
    }

    std::string input_path = argv[3];
    std::string output_path = argv[4];

    cv::Mat original_image = cv::imread(input_path);
    if (original_image.empty()) {
        std::cerr << "Could not read image from " << input_path << "\n";
        return 1;
    }

    int original_width = original_image.cols;
    int original_height = original_image.rows;

    torch::Tensor input_tensor = preprocessFrame(original_image);

    torch::jit::script::Module model;
    try {
        model = torch::jit::load("yolo11n.torchscript");
    }
    catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return 1;
    }

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "Using CUDA device" << std::endl;
    } else if (torch::mps::is_available()) {
        device = torch::Device(torch::kMPS);
        std::cout << "Using MPS device" << std::endl;
    } else {
        std::cout << "Using CPU device" << std::endl;
    }

    model.to(device);
    model.eval();

    torch::NoGradGuard no_grad;
    torch::Tensor      output;
    try {
        output = model.forward({input_tensor.to(device)}).toTensor().to(torch::kCPU);
    }
    catch (const c10::Error &e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Output tensor shape: " << output.sizes() << std::endl;

    if (output.dim() == 3 && output.size(0) == 1 && output.size(1) == 84 && output.size(2) == 8400) {
        output = output.squeeze(0).transpose(0, 1);
        std::cout << "Transposed output shape: " << output.sizes() << std::endl;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      class_ids;
    const float           input_width = 640.0f;
    const float           input_height = 640.0f;

    int num_detections = output.size(0);
    std::cout << "Processing " << num_detections << " potential detections" << std::endl;

    for (int i = 0; i < num_detections; ++i) {
        auto class_scores = output[i].slice(0, 4, 84);
        auto [max_score, max_idx] = torch::max(class_scores, 0);
        float confidence = max_score.item<float>();

        if (confidence >= confidence_threshold) {
            int class_id = max_idx.item<int>();

            float center_x = output[i][0].item<float>();
            float center_y = output[i][1].item<float>();
            float width = output[i][2].item<float>();
            float height = output[i][3].item<float>();

            std::cout << "Raw coords: " << center_x << ", " << center_y << ", " << width << ", " << height << std::endl;

            float x = (center_x / input_width) * original_width;
            float y = (center_y / input_height) * original_height;
            float w = (width / input_width) * original_width;
            float h = (height / input_height) * original_height;

            float left = x - w / 2;
            float top = y - h / 2;

            left = std::max(0.0f, left);
            top = std::max(0.0f, top);
            w = std::min(w, static_cast<float>(original_width - left));
            h = std::min(h, static_cast<float>(original_height - top));

            cv::Rect box(static_cast<int>(left), static_cast<int>(top), static_cast<int>(w), static_cast<int>(h));

            boxes.push_back(box);
            scores.push_back(confidence);
            class_ids.push_back(class_id);

            std::cout << "Detection: class=" << class_id << " (" << class_names[class_id] << "), conf=" << std::fixed
                      << std::setprecision(2) << confidence << std::endl;
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, non_max_suppression_threshold, indices);

    std::cout << "After NMS: " << indices.size() << " detections" << std::endl;

    for (int idx: indices) {
        cv::Rect box = boxes[idx];
        int      class_id = class_ids[idx];
        float    score = scores[idx];

        // Get unique color for this class
        cv::Scalar color = getColorForClass(class_id);

        // Draw bounding box with class-specific color
        cv::rectangle(original_image, box, color, 2);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << score;
        std::string score_str = ss.str();
        std::string label = class_names[class_id] + ": " + score_str;

        int      baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);

        // Draw label background with the same class-specific color
        cv::rectangle(
          original_image, cv::Point(box.x, box.y - text_size.height - 5), cv::Point(box.x + text_size.width, box.y),
          color, -1
        );

        // Draw text in black for contrast
        cv::putText(
          original_image, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2
        );

        std::cout << "Drew box for " << label << " at " << box.x << "," << box.y << " size " << box.width << "x"
                  << box.height << std::endl;
    }

    if (!cv::imwrite(output_path, original_image)) {
        std::cerr << "Failed to save output image to " << output_path << "\n";
        return 1;
    }

    std::cout << "Output image saved to " << output_path << "\n";
    std::cout << "Found " << indices.size() << " objects" << std::endl;

    return 0;
}