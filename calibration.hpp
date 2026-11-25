#pragma once

#include <opencv2/core.hpp>
#include <string>

class Calibration {
public:
    bool load(const std::string& path);

    cv::Mat K;
    cv::Mat D;
    cv::Matx33d R_cam2gimbal = cv::Matx33d::eye();
    cv::Vec3d t_cam2gimbal{0, 0, 0};
    cv::Matx33d R_gimbal2imu = cv::Matx33d::eye();
    int width = 0;
    int height = 0;

    cv::Vec3d cameraToGimbal(const cv::Vec3d& p) const;
    static cv::Matx33d quatToMatrix(double qw, double qx, double qy, double qz);
};
