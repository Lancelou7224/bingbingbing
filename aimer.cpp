#include "aimer.hpp"

#include <algorithm>
#include <cmath>

namespace {

double wrap180(double a) {
    while (a > 180.0) a -= 360.0;
    while (a < -180.0) a += 360.0;
    return a;
}

}  // namespace

AimState Aimer::update(const Track* track, double dt) {
    AimState state;
    if (!cfg_.enable || !track || track->hits < 3) {
        fire_count_ = 0;
        return state;
    }
    if (dt <= 1e-5) dt = 1.0 / 166.0;

    const cv::Vec3d p = track->xyz_gimbal + track->vel_gimbal * cfg_.prediction_time;
    if (p[0] < 0.2) {
        fire_count_ = 0;
        return state;
    }

    const double distance = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    if (distance < 0.2 || distance > 15.0) {
        fire_count_ = 0;
        return state;
    }

    double yaw = std::atan2(p[1], p[0]) * 180.0 / CV_PI;
    double pitch = std::atan2(p[2], std::sqrt(p[0] * p[0] + p[1] * p[1])) * 180.0 / CV_PI;

    // 弹道下坠补偿，近距离影响很小，远距离大约 0.1~0.5 度
    const double flight = distance / std::max(1.0f, cfg_.bullet_speed);
    const double drop = 0.5 * cfg_.gravity * flight * flight;
    pitch += std::atan2(drop, distance) * 180.0 / CV_PI;

    yaw += cfg_.yaw_offset;
    pitch += cfg_.pitch_offset;
    yaw = wrap180(yaw);

    if (!initialized_) {
        gimbal_yaw_ = yaw;
        gimbal_pitch_ = pitch;
        initialized_ = true;
    } else {
        const double max_step = cfg_.max_gimbal_speed * 180.0 / CV_PI * dt;
        const double dy = wrap180(yaw - gimbal_yaw_);
        const double dp = pitch - gimbal_pitch_;
        gimbal_yaw_ += std::max(-max_step, std::min(max_step, dy));
        gimbal_pitch_ += std::max(-max_step, std::min(max_step, dp));
    }

    state.valid = true;
    state.target_yaw = yaw;
    state.target_pitch = pitch;
    state.gimbal_yaw = gimbal_yaw_;
    state.gimbal_pitch = gimbal_pitch_;
    state.yaw_err = wrap180(yaw - gimbal_yaw_);
    state.pitch_err = pitch - gimbal_pitch_;

    const bool aligned = std::fabs(state.yaw_err) < cfg_.fire_yaw_tol &&
                         std::fabs(state.pitch_err) < cfg_.fire_pitch_tol;
    if (aligned && track->misses == 0) {
        fire_count_++;
    } else {
        fire_count_ = 0;
    }
    state.fire = fire_count_ >= cfg_.fire_frames;
    return state;
}
