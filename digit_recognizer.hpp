#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>

class DigitRecognizer {
public:
    bool load(const std::string& model_path);
    bool ready() const { return ready_; }

    // armor_warp 为透视变换后的装甲板图像，比例约 135:56
    bool recognize(const cv::Mat& armor_warp, int& digit, float& confidence);

private:
    cv::dnn::Net net_;
    bool ready_ = false;
};
