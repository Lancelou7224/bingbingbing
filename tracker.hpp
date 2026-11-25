#pragma once

#include "types.hpp"
#include <opencv2/core.hpp>
#include <vector>

struct Track {
    int id = -1;
    cv::Point2f center;
    cv::Point2f velocity;
    cv::Rect2f bbox;
    cv::Vec3d xyz_cam{0, 0, 0};
    cv::Vec3d xyz_gimbal{0, 0, 0};
    cv::Vec3d vel_cam{0, 0, 0};
    cv::Vec3d vel_gimbal{0, 0, 0};
    int hits = 0;
    int misses = 0;
    int age = 0;
    int digit = -1;
    int type = 0;  // 0 小装甲板, 1 大装甲板
    bool matched = false;
    int votes[10] = {0};
    int vote_total = 0;
    float last_conf = 0.f;

    bool confirmed(int min_hits) const { return hits >= min_hits; }
    cv::Point2f predictCenter(double dt) const { return center + velocity * static_cast<float>(dt); }
};

class Tracker {
public:
    explicit Tracker(const TrackerConfig& cfg) : cfg_(cfg) {}

    std::vector<Track> update(std::vector<Armor>& detections, double dt);
    const std::vector<Track>& tracks() const { return tracks_; }
    const Track* best() const;
    cv::Rect bestRoi(const cv::Size& frame_size, float padding) const;

private:
    void updateDigit(Track& track, const Armor& armor);
    int rank(const Track& track) const;

    TrackerConfig cfg_;
    std::vector<Track> tracks_;
    int next_id_ = 0;
};
