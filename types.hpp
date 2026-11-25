#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct LightBar {
    cv::Point2f top;
    cv::Point2f bottom;
    cv::Point2f center;
    float length = 0.f;
    float width = 0.f;
    float ratio = 0.f;
    float tilt = 0.f;
    float area = 0.f;
};

struct Armor {
    std::vector<cv::Point2f> corners;  // TL, TR, BR, BL
    cv::Point2f center;
    cv::Rect2f bbox;
    int color = 0;       // 0 red, 1 blue
    int type = 0;        // 0 small, 1 large
    float ratio = 0.f;
    float score = 0.f;
    float reproj = 0.f;
    cv::Vec3d xyz_cam{0, 0, 0};
    cv::Vec3d xyz_gimbal{0, 0, 0};
    int digit = -1;
    float digit_conf = 0.f;
    int track_id = -1;

    bool valid() const { return !corners.empty() && xyz_cam[2] > 0; }
};

struct DetectorConfig {
    int enemy_color = 0;  // 0 red, 1 blue
    int min_saturation = 120;
    int min_value = 100;
    int red_hue_max = 28;
    int blue_hue_min = 90;
    int blue_hue_max = 135;

    int morph_size = 3;
    float min_light_area = 16.f;
    float min_light_length = 10.f;
    float min_light_ratio = 1.4f;
    float max_light_ratio = 15.f;
    float max_tilt = 45.f;

    float min_pair_ratio = 1.3f;
    float max_pair_ratio = 4.6f;
    float max_angle_diff = 22.f;
    float max_y_diff_ratio = 0.7f;
    float min_overlap_ratio = 0.45f;
    float min_length_ratio = 0.6f;
    float min_armor_width = 24.f;
    float min_distance = 0.3f;
    float max_distance = 8.0f;
    float large_ratio_thresh = 3.3f;
    float max_reproj = 12.f;
    int max_armors = 4;

    int scale = 1;          // 1 = full resolution, 2 = half resolution
    bool use_roi = false;
    float roi_padding = 0.6f;
};

struct TrackerConfig {
    int min_hits = 3;
    int max_misses = 8;
    float assoc_dist_ratio = 1.5f;
    float velocity_smooth = 0.6f;
    int digit_vote_min = 3;
    float digit_conf_thresh = 0.45f;
};

struct AimerConfig {
    bool enable = true;
    float bullet_speed = 20.0f;
    float gravity = 9.8f;
    float yaw_offset = 0.f;
    float pitch_offset = 0.f;
    float max_gimbal_speed = 8.0f;  // rad/s
    float fire_yaw_tol = 1.5f;      // degree
    float fire_pitch_tol = 2.0f;    // degree
    int fire_frames = 2;
    float prediction_time = 0.03f;  // s
};

struct AppConfig {
    std::string dataset;
    std::string output = "output/demo.mp4";
    std::string model = "models/mnist-12.onnx";
    int out_width = 1280;
    int out_height = 720;
    int out_fps = 60;
    int stride = 1;
    int max_frames = 0;

    bool benchmark = false;
    bool draw = true;
    bool use_digit = true;
    bool fast = false;

    DetectorConfig detector;
    TrackerConfig tracker;
    AimerConfig aimer;
};
