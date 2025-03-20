#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "preprocess_frame.hpp"

torch::Tensor preprocess_frame(cv::Mat &frame)
{
    cv::Mat img;
    cv::resize(frame, img, cv::Size(640, 640));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    torch::Tensor tensor_image = torch::from_blob(img.data, {1, 640, 640, 3}, torch::kFloat32);
    tensor_image = tensor_image.permute({0, 3, 1, 2});
    return tensor_image.clone();
}