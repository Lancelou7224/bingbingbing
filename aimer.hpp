#pragma once

#include "tracker.hpp"
#include "types.hpp"

struct AimState {
    bool valid = false;
    double target_yaw = 0.0;
    double target_pitch = 0.0;
    double gimbal_yaw = 0.0;
    double gimbal_pitch = 0.0;
    double yaw_err = 0.0;
    double pitch_err = 0.0;
    bool fire = false;
};

class Aimer {
public:
    explicit Aimer(const AimerConfig& cfg) : cfg_(cfg) {}

    AimState update(const Track* track, double dt);

private:
    AimerConfig cfg_;
    bool initialized_ = false;
    double gimbal_yaw_ = 0.0;
    double gimbal_pitch_ = 0.0;
    int fire_count_ = 0;
};
