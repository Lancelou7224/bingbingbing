#include "tracker.hpp"

#include <algorithm>
#include <cmath>

namespace {

float bboxSize(const cv::Rect2f& b) { return std::max(b.width, b.height); }

// 跟踪速度限幅，防止个别帧的三维解跳变被当成高速运动
cv::Vec3d clampSpeed(const cv::Vec3d& v, double max_speed) {
    const double n = cv::norm(v);
    if (n < 1e-9 || n <= max_speed) return v;
    return v * (max_speed / n);
}

}  // namespace

std::vector<Track> Tracker::update(std::vector<Armor>& detections, double dt) {
    if (dt <= 1e-5) dt = 1.0 / 166.0;

    for (auto& t : tracks_) {
        t.center = t.predictCenter(dt);
        t.matched = false;
    }

    struct Match {
        float cost;
        int track;
        int det;
    };
    std::vector<Match> matches;
    for (int ti = 0; ti < static_cast<int>(tracks_.size()); ++ti) {
        const Track& t = tracks_[ti];
        const float max_dist = std::max(40.f, cfg_.assoc_dist_ratio * bboxSize(t.bbox));
        const double depth = cv::norm(t.xyz_cam);
        for (int di = 0; di < static_cast<int>(detections.size()); ++di) {
            const float d2 = cv::norm(t.center - detections[di].center);
            if (d2 > max_dist) continue;

            // 两个目标在原图上可能挨得很近，用三维距离再约束一下，避免轨迹互换
            const double d3 = cv::norm(t.xyz_cam - detections[di].xyz_cam);
            if (d3 > std::max(0.6, 0.5 * depth)) continue;

            const float cost = d2 / max_dist + static_cast<float>(d3 / std::max(1.0, depth));
            matches.push_back({cost, ti, di});
        }
    }
    std::sort(matches.begin(), matches.end(),
              [](const Match& a, const Match& b) { return a.cost < b.cost; });

    std::vector<int> track_match(tracks_.size(), -1);
    std::vector<int> det_match(detections.size(), -1);
    for (const auto& m : matches) {
        if (track_match[m.track] >= 0 || det_match[m.det] >= 0) continue;
        track_match[m.track] = m.det;
        det_match[m.det] = m.track;
    }

    for (int ti = 0; ti < static_cast<int>(tracks_.size()); ++ti) {
        Track& t = tracks_[ti];
        const int di = track_match[ti];
        if (di < 0) {
            t.misses++;
            t.age++;
            // 只做短时间外推，丢失太久就停在最后的位置
            if (t.misses <= 3) {
                t.xyz_cam += t.vel_cam * dt;
                t.xyz_gimbal += t.vel_gimbal * dt;
                t.bbox.x += t.velocity.x * static_cast<float>(dt);
                t.bbox.y += t.velocity.y * static_cast<float>(dt);
            }
            continue;
        }

        Armor& a = detections[di];
        const cv::Point2f meas_vel = (a.center - t.center) / static_cast<float>(dt);
        t.velocity = t.velocity * cfg_.velocity_smooth + meas_vel * (1.f - cfg_.velocity_smooth);

        const cv::Vec3d meas_cam = (a.xyz_cam - t.xyz_cam) / dt;
        const cv::Vec3d meas_gimbal = (a.xyz_gimbal - t.xyz_gimbal) / dt;
        t.vel_cam = clampSpeed(
            t.vel_cam * cfg_.velocity_smooth + meas_cam * (1.f - cfg_.velocity_smooth), 12.0);
        t.vel_gimbal = clampSpeed(
            t.vel_gimbal * cfg_.velocity_smooth + meas_gimbal * (1.f - cfg_.velocity_smooth), 12.0);

        t.center = a.center;
        t.bbox = a.bbox;
        t.xyz_cam = a.xyz_cam;
        t.xyz_gimbal = a.xyz_gimbal;
        t.type = a.type;
        t.hits++;
        t.misses = 0;
        t.age++;
        t.matched = true;
        a.track_id = t.id;
        updateDigit(t, a);
    }

    for (int di = 0; di < static_cast<int>(detections.size()); ++di) {
        if (det_match[di] >= 0) continue;
        Track t;
        t.id = next_id_++;
        t.center = detections[di].center;
        t.bbox = detections[di].bbox;
        t.xyz_cam = detections[di].xyz_cam;
        t.xyz_gimbal = detections[di].xyz_gimbal;
        t.type = detections[di].type;
        t.hits = 1;
        t.age = 1;
        t.matched = true;
        detections[di].track_id = t.id;
        updateDigit(t, detections[di]);
        tracks_.push_back(t);
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [this](const Track& t) { return t.misses > cfg_.max_misses; }),
                  tracks_.end());

    return tracks_;
}

void Tracker::updateDigit(Track& track, const Armor& armor) {
    if (armor.digit < 0 || armor.digit > 9) return;
    if (armor.digit_conf < cfg_.digit_conf_thresh) return;

    track.votes[armor.digit]++;
    track.vote_total++;
    track.last_conf = armor.digit_conf;

    if (track.vote_total < cfg_.digit_vote_min) return;
    int best = 0;
    for (int i = 1; i < 10; ++i) {
        if (track.votes[i] > track.votes[best]) best = i;
    }
    if (track.votes[best] >= cfg_.digit_vote_min && track.votes[best] * 2 >= track.vote_total) {
        track.digit = best;
    }
}

int Tracker::rank(const Track& track) const {
    int score = 0;
    if (track.matched) score += 10000;
    if (track.digit >= 0) score += 2000;
    if (track.digit >= 0 && track.type == 0) score += 1000;
    score += track.hits * 4 - track.misses * 8;
    return score;
}

const Track* Tracker::best() const {
    const Track* best = nullptr;
    for (const auto& t : tracks_) {
        if (t.misses > 0 && t.hits < cfg_.min_hits) continue;
        if (!best || rank(t) > rank(*best)) best = &t;
    }
    return best;
}

cv::Rect Tracker::bestRoi(const cv::Size& frame_size, float padding) const {
    const Track* t = best();
    if (!t || t->hits < 2) return cv::Rect();
    const float pad = padding * bboxSize(t->bbox);
    const int x = static_cast<int>(t->bbox.x - pad);
    const int y = static_cast<int>(t->bbox.y - pad);
    const int w = static_cast<int>(t->bbox.width + 2 * pad);
    const int h = static_cast<int>(t->bbox.height + 2 * pad);
    return cv::Rect(x, y, w, h) & cv::Rect(0, 0, frame_size.width, frame_size.height);
}
