#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/script.h>
#include <torch/torch.h>

struct Camera
{
    int camera_index;
    int channels;
    int width;
    int height;
};

torch::Tensor preprocessFrame(cv::Mat &frame)
{
    cv::Mat img;

    // Resize to match YOLO input size
    cv::resize(frame, img, cv::Size(640, 640));

    // Convert BGR (OpenCV default) to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Convert image to float32 and normalize to [0,1]
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // Convert OpenCV Mat to Torch Tensor
    torch::Tensor tensor_image = torch::from_blob(img.data, {1, 640, 640, 3}, torch::kFloat32);

    // Permute to match PyTorch format (C, H, W)
    tensor_image = tensor_image.permute({0, 3, 1, 2});

    return tensor_image.clone(); // Clone to ensure memory is properly managed
}

int main()
{
    // input shape : (1, 3, 640, 640)
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
        std::cout << "not capturing\n";
        return -1;
    } else {
        // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        // cap.set(cv::CAP_PROP_FPS, 45);
        // cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }

    cv::namedWindow("highest resolution camera", cv::WINDOW_NORMAL);
    cv::Mat frame;

    std::cout << "testing frame capture\n";

    int empty_frames = 0;

    for (int i = 0; i < 100; i++) {
        cap >> frame;
        empty_frames += static_cast<int>(frame.empty());
    }

    if (empty_frames > 0) {
        std::cerr << "empty frames captured : " << empty_frames << "\n";
        return 2;
    }

    std::cout << "video caputre started\n";

    while (true) {
        cap >> frame;

        // Convert frame to tensor
        torch::Tensor input_tensor = preprocessFrame(frame);

        input_tensor = input_tensor.to(device);

        torch::Tensor output = model.forward({input_tensor}).toTensor();

        std::cout << "Inference done! Output shape: " << output.sizes() << std::endl;

        cv::imshow("highest resolution camera", frame);

        if (cv::waitKey(1) >= 0) {
            break; // exit on key press
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}