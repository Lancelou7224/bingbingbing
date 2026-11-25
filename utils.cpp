#include "utils.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>

double nowMs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

namespace {

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> out;
    std::string item;
    std::istringstream ss(line);
    while (std::getline(ss, item, ',')) out.push_back(item);
    return out;
}

}  // namespace

std::vector<double> loadTimestamps(const std::string& path) {
    std::vector<double> out;
    std::ifstream in(path);
    if (!in) return out;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto parts = splitCsv(line);
        if (parts.size() < 4) continue;
        try {
            out.push_back(std::stod(parts[3]));
        } catch (...) {
        }
    }
    return out;
}

std::vector<cv::Vec4d> loadQuaternions(const std::string& path) {
    std::vector<cv::Vec4d> out;
    std::ifstream in(path);
    if (!in) return out;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto parts = splitCsv(line);
        if (parts.size() < 7) continue;
        try {
            out.emplace_back(std::stod(parts[3]), std::stod(parts[4]), std::stod(parts[5]),
                             std::stod(parts[6]));
        } catch (...) {
        }
    }
    return out;
}

cv::Mat makeArmorWarp(const cv::Mat& frame, const Armor& armor, int width, int height) {
    if (armor.corners.size() != 4) return {};
    const std::vector<cv::Point2f> dst = {
        {0.f, 0.f},
        {static_cast<float>(width - 1), 0.f},
        {static_cast<float>(width - 1), static_cast<float>(height - 1)},
        {0.f, static_cast<float>(height - 1)}};
    const cv::Mat M = cv::getPerspectiveTransform(armor.corners, dst);
    cv::Mat warp;
    cv::warpPerspective(frame, warp, M, cv::Size(width, height));
    return warp;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double idx = p * static_cast<double>(values.size() - 1);
    const int lo = static_cast<int>(std::floor(idx));
    const int hi = static_cast<int>(std::ceil(idx));
    if (lo == hi) return values[lo];
    const double w = idx - lo;
    return values[lo] * (1.0 - w) + values[hi] * w;
}
