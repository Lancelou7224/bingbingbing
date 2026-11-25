#include "calibration.hpp"

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>

namespace {

cv::Mat readMat(const YAML::Node& node, int rows, int cols) {
    if (!node || !node.IsSequence() || static_cast<int>(node.size()) != rows * cols) {
        throw std::runtime_error("calibration: unexpected matrix size");
    }
    cv::Mat m(rows, cols, CV_64F);
    int k = 0;
    for (const auto& v : node) {
        m.at<double>(k / cols, k % cols) = v.as<double>();
        k++;
    }
    return m;
}

cv::Matx33d matToMatx33d(const cv::Mat& m) {
    cv::Matx33d out;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out(r, c) = m.at<double>(r, c);
    return out;
}

}  // namespace

bool Calibration::load(const std::string& path) {
    try {
        YAML::Node root = YAML::LoadFile(path);
        K = readMat(root["camera_matrix"], 3, 3);
        D = readMat(root["distort_coeffs"], 1, 5);
        R_cam2gimbal = matToMatx33d(readMat(root["R_camera2gimbal"], 3, 3));
        cv::Mat t = readMat(root["t_camera2gimbal"], 3, 1);
        t_cam2gimbal = cv::Vec3d(t.at<double>(0), t.at<double>(1), t.at<double>(2));
        R_gimbal2imu = matToMatx33d(readMat(root["R_gimbal2imubody"], 3, 3));
        if (root["image_width"]) width = root["image_width"].as<int>();
        if (root["image_height"]) height = root["image_height"].as<int>();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "load calibration failed: " << e.what() << "\n";
        return false;
    }
}

cv::Vec3d Calibration::cameraToGimbal(const cv::Vec3d& p) const {
    return R_cam2gimbal * p + t_cam2gimbal;
}

cv::Matx33d Calibration::quatToMatrix(double qw, double qx, double qy, double qz) {
    const double n = qw * qw + qx * qx + qy * qy + qz * qz;
    if (n < 1e-12) return cv::Matx33d::eye();
    const double s = 2.0 / n;
    return cv::Matx33d(
        1 - s * (qy * qy + qz * qz), s * (qx * qy - qw * qz), s * (qx * qz + qw * qy),
        s * (qx * qy + qw * qz), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qw * qx),
        s * (qx * qz - qw * qy), s * (qy * qz + qw * qx), 1 - s * (qx * qx + qy * qy));
}
