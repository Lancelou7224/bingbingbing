#pragma once

#include "calibration.hpp"
#include "types.hpp"
#include <opencv2/core.hpp>
#include <vector>

class Detector {
public:
    Detector(const DetectorConfig& cfg, const Calibration& calib);

    std::vector<Armor> detect(const cv::Mat& bgr, const cv::Rect& roi = cv::Rect());

private:
    std::vector<LightBar> findLightBars(const cv::Mat& bgr) const;
    std::vector<Armor> pairArmors(const std::vector<LightBar>& bars) const;
    bool solvePnp(Armor& armor) const;

    DetectorConfig cfg_;
    const Calibration& calib_;
};
