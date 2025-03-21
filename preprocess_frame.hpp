#ifndef MRDCVLSC_OPENCV_CMAKE_TEMPLATE_PREPROCESS_FRAME_HPP
#define MRDCVLSC_OPENCV_CMAKE_TEMPLATE_PREPROCESS_FRAME_HPP

#include <opencv2/core.hpp>
#include <torch/torch.h>

torch::Tensor preprocess_frame(cv::Mat &frame, int w, int h, torch::ScalarType data_type);

#endif