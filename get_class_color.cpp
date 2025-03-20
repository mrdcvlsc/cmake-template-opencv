#include <opencv2/imgproc.hpp>

#include "get_class_color.hpp"

cv::Scalar get_class_color(int class_id)
{
    // use HSV color space where Hue varies by class ID
    float   hue = std::fmod(class_id * 137.f, 360.f); // 137 is a prime number for good distribution
    cv::Mat hsv(1, 1, CV_32FC3);
    hsv.at<cv::Vec3f>(0, 0) = cv::Vec3f(hue, 1.f, 1.f); // full saturation and value
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3f color = bgr.at<cv::Vec3f>(0, 0);
    return cv::Scalar(color[0] * 255, color[1] * 255, color[2] * 255);
}