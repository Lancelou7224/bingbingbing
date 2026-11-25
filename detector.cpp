#include "detector.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace {

struct Fragment {
    std::vector<cv::Point2f> pts;
    cv::Point2f center;
    cv::Point2f axis;
    float length = 0.f;
    float width = 0.f;
    float area = 0.f;
    float lo = 0.f;
    float hi = 0.f;
};

bool fitFragment(const std::vector<cv::Point2f>& pts, float area, Fragment& f) {
    if (pts.size() < 4) return false;

    cv::Mat data(static_cast<int>(pts.size()), 2, CV_32F);
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        data.at<float>(i, 0) = pts[i].x;
        data.at<float>(i, 1) = pts[i].y;
    }

    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    f.center = cv::Point2f(pca.mean.at<float>(0), pca.mean.at<float>(1));
    f.axis = cv::Point2f(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
    const float n = std::sqrt(f.axis.x * f.axis.x + f.axis.y * f.axis.y);
    if (n < 1e-6f) return false;
    f.axis.x /= n;
    f.axis.y /= n;

    f.lo = 1e9f;
    f.hi = -1e9f;
    for (const auto& p : pts) {
        const float proj = (p - f.center).dot(f.axis);
        f.lo = std::min(f.lo, proj);
        f.hi = std::max(f.hi, proj);
    }

    f.pts = pts;
    f.length = f.hi - f.lo;
    f.area = area;
    f.width = area / std::max(f.length, 1e-3f);
    return true;
}

// 灯条被保险杠、数字底板挡住后会断成几段，先按共线关系把碎片并回去
bool mergeable(const Fragment& a, const Fragment& b) {
    if (std::fabs(a.axis.dot(b.axis)) < 0.978f) return false;  // 夹角超过 12°

    const cv::Point2f d = b.center - a.center;
    const float perp = std::fabs(d.x * -a.axis.y + d.y * a.axis.x);
    if (perp > std::max(3.f, 0.8f * (a.width + b.width))) return false;

    const float gap = std::fabs(d.dot(a.axis)) - (a.length + b.length) * 0.5f;
    return gap < 0.6f * std::max(a.length, b.length) + 4.f;
}

float angleOfBar(const LightBar& b) {
    // 灯条长轴相对竖直方向的夹角，向右倾为正
    return static_cast<float>(std::atan2(b.top.x - b.bottom.x, b.bottom.y - b.top.y) * 180.0 /
                              CV_PI);
}

float normalizeAngle(float a) {
    while (a > 180.f) a -= 360.f;
    while (a < -180.f) a += 360.f;
    return a;
}

}  // namespace

Detector::Detector(const DetectorConfig& cfg, const Calibration& calib) : cfg_(cfg), calib_(calib) {}

std::vector<LightBar> Detector::findLightBars(const cv::Mat& bgr) const {
    std::vector<LightBar> bars;
    if (bgr.empty()) return bars;

    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    const int s = std::max(0, cfg_.min_saturation);
    const int v = std::max(0, cfg_.min_value);
    if (cfg_.enemy_color == 0) {
        cv::Mat m1, m2;
        cv::inRange(hsv, cv::Scalar(0, s, v), cv::Scalar(cfg_.red_hue_max, 255, 255), m1);
        cv::inRange(hsv, cv::Scalar(160, s, v), cv::Scalar(180, 255, 255), m2);
        mask = m1 | m2;
    } else {
        cv::inRange(hsv, cv::Scalar(cfg_.blue_hue_min, s, v),
                    cv::Scalar(cfg_.blue_hue_max, 255, 255), mask);
    }

    const int k = std::max(1, cfg_.morph_size);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Fragment> frags;
    for (const auto& contour : contours) {
        if (contour.size() < 4) continue;
        const float area = static_cast<float>(cv::contourArea(contour));
        if (area < cfg_.min_light_area) continue;

        std::vector<cv::Point2f> pts;
        pts.reserve(contour.size());
        for (const auto& p : contour) pts.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));

        Fragment f;
        if (!fitFragment(pts, area, f)) continue;
        if (f.length < cfg_.min_light_length * 0.4f) continue;
        if (f.width < 1.0f) continue;
        frags.push_back(std::move(f));
    }

    std::vector<int> parent(frags.size());
    for (size_t i = 0; i < parent.size(); ++i) parent[i] = static_cast<int>(i);
    auto findRoot = [&parent](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    for (size_t i = 0; i < frags.size(); ++i) {
        for (size_t j = i + 1; j < frags.size(); ++j) {
            if (!mergeable(frags[i], frags[j])) continue;
            const int ri = findRoot(static_cast<int>(i));
            const int rj = findRoot(static_cast<int>(j));
            if (ri != rj) parent[rj] = ri;
        }
    }

    std::vector<std::vector<cv::Point2f>> groups(frags.size());
    std::vector<float> group_area(frags.size(), 0.f);
    for (size_t i = 0; i < frags.size(); ++i) {
        const int root = findRoot(static_cast<int>(i));
        auto& g = groups[root];
        g.insert(g.end(), frags[i].pts.begin(), frags[i].pts.end());
        group_area[root] += frags[i].area;
    }

    for (size_t gi = 0; gi < groups.size(); ++gi) {
        const auto& pts = groups[gi];
        if (pts.empty()) continue;
        Fragment f;
        if (!fitFragment(pts, group_area[gi], f)) continue;

        const float ratio = f.length / std::max(f.width, 1e-3f);
        if (f.length < cfg_.min_light_length) continue;
        if (f.width < 1.5f) continue;
        if (ratio < cfg_.min_light_ratio || ratio > cfg_.max_light_ratio) continue;

        LightBar bar;
        bar.center = f.center;
        bar.top = f.center + f.axis * f.hi;
        bar.bottom = f.center + f.axis * f.lo;
        if (bar.top.y > bar.bottom.y) std::swap(bar.top, bar.bottom);
        bar.length = f.length;
        bar.width = f.width;
        bar.ratio = ratio;
        bar.area = f.area;
        bar.tilt = std::fabs(angleOfBar(bar));
        if (bar.tilt > cfg_.max_tilt) continue;

        bars.push_back(bar);
    }

    std::sort(bars.begin(), bars.end(),
              [](const LightBar& a, const LightBar& b) { return a.center.x < b.center.x; });
    return bars;
}

std::vector<Armor> Detector::pairArmors(const std::vector<LightBar>& bars) const {
    struct Candidate {
        float score = 0.f;
        float ratio = 0.f;
        int left_idx = -1;
        int right_idx = -1;
        LightBar left;
        LightBar right;
        cv::Point2f center;
        float height = 0.f;
    };

    std::vector<Candidate> candidates;
    for (size_t i = 0; i < bars.size(); ++i) {
        for (size_t j = i + 1; j < bars.size(); ++j) {
            const LightBar& a = bars[i];
            const LightBar& b = bars[j];

            // 两根灯条之间如果还有同高度的灯条，说明它们不是一对
            bool has_between = false;
            const float mid_y = (a.center.y + b.center.y) * 0.5f;
            for (size_t k = 0; k < bars.size(); ++k) {
                if (k == i || k == j) continue;
                const LightBar& c = bars[k];
                if (c.center.x > a.center.x + 4.f && c.center.x < b.center.x - 4.f &&
                    std::fabs(c.center.y - mid_y) < std::max(a.length, b.length) * 0.75f) {
                    has_between = true;
                    break;
                }
            }
            if (has_between) continue;

            const float l1 = a.length, l2 = b.length;
            const float h = (l1 + l2) * 0.5f;
            if (h < 1.f) continue;
            if (std::min(l1, l2) / std::max(l1, l2) < cfg_.min_length_ratio) continue;

            const float angle_diff = std::fabs(normalizeAngle(angleOfBar(a) - angleOfBar(b)));
            if (angle_diff > cfg_.max_angle_diff) continue;

            const float dx = b.center.x - a.center.x;
            if (dx < cfg_.min_armor_width) continue;
            const float ratio = dx / h;
            if (ratio < cfg_.min_pair_ratio || ratio > cfg_.max_pair_ratio) continue;

            if (std::fabs(a.top.y - b.top.y) > cfg_.max_y_diff_ratio * h) continue;
            if (std::fabs(a.bottom.y - b.bottom.y) > cfg_.max_y_diff_ratio * h) continue;
            if (std::fabs(a.center.y - b.center.y) > 0.55f * h) continue;

            const float overlap = std::min(a.bottom.y, b.bottom.y) - std::max(a.top.y, b.top.y);
            if (overlap < cfg_.min_overlap_ratio * h) continue;

            // 小装甲板 135x56，大装甲板 225x56，两个比例都算合理
            const float small_err = std::fabs(ratio - 2.41f);
            const float large_err = std::fabs(ratio - 4.02f);
            const float ratio_score = std::max(0.f, 1.f - std::min(small_err, large_err) / 1.15f);
            const float angle_score = 1.f - std::min(1.f, angle_diff / cfg_.max_angle_diff);
            const float len_score = std::min(l1, l2) / std::max(l1, l2);

            Candidate c;
            c.score = 0.45f * ratio_score + 0.30f * angle_score + 0.25f * len_score;
            c.ratio = ratio;
            c.left_idx = static_cast<int>(i);
            c.right_idx = static_cast<int>(j);
            c.left = a;
            c.right = b;
            c.height = h;
            c.center = cv::Point2f((a.center.x + b.center.x) * 0.5f,
                                   (a.top.y + b.top.y + a.bottom.y + b.bottom.y) * 0.25f);
            candidates.push_back(c);
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    std::vector<Candidate> kept;
    std::vector<bool> used(bars.size(), false);
    for (const auto& c : candidates) {
        // 一根灯条只可能属于一块装甲板，避免把并排两台车的灯条凑成一对
        if (used[c.left_idx] || used[c.right_idx]) continue;

        bool ok = true;
        for (const auto& k : kept) {
            if (cv::norm(c.center - k.center) < std::max(c.height, k.height) * 0.9f) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;

        kept.push_back(c);
        used[c.left_idx] = true;
        used[c.right_idx] = true;
        if (static_cast<int>(kept.size()) >= cfg_.max_armors) break;
    }

    std::vector<Armor> armors;
    for (const auto& c : kept) {
        Armor armor;
        armor.corners = {c.left.top, c.right.top, c.right.bottom, c.left.bottom};
        armor.center = c.center;
        armor.bbox = cv::boundingRect(armor.corners);
        armor.color = cfg_.enemy_color;
        armor.ratio = c.ratio;
        armor.score = c.score;
        armor.type = (cfg_.large_ratio_thresh > 0.f && c.ratio > cfg_.large_ratio_thresh) ? 1 : 0;
        if (solvePnp(armor)) armors.push_back(armor);
    }
    return armors;
}

bool Detector::solvePnp(Armor& armor) const {
    if (armor.corners.size() != 4) return false;

    const double w = (armor.type == 0) ? 0.135 : 0.225;
    const double h = 0.056;
    const std::vector<cv::Point3f> object_points = {
        {-static_cast<float>(w / 2), -static_cast<float>(h / 2), 0.f},
        {static_cast<float>(w / 2), -static_cast<float>(h / 2), 0.f},
        {static_cast<float>(w / 2), static_cast<float>(h / 2), 0.f},
        {-static_cast<float>(w / 2), static_cast<float>(h / 2), 0.f}};

    cv::Mat rvec, tvec;
    try {
        if (!cv::solvePnP(object_points, armor.corners, calib_.K, calib_.D, rvec, tvec, false,
                          cv::SOLVEPNP_IPPE)) {
            return false;
        }
    } catch (const cv::Exception&) {
        return false;
    }

    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec, tvec, calib_.K, calib_.D, projected);
    float err = 0.f;
    for (size_t i = 0; i < projected.size(); ++i) err += cv::norm(projected[i] - armor.corners[i]);
    err /= static_cast<float>(projected.size());
    armor.reproj = err;
    if (err > cfg_.max_reproj) return false;

    armor.xyz_cam = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    const double dist = cv::norm(armor.xyz_cam);
    if (dist < cfg_.min_distance || dist > cfg_.max_distance) return false;
    armor.xyz_gimbal = calib_.cameraToGimbal(armor.xyz_cam);
    return true;
}

std::vector<Armor> Detector::detect(const cv::Mat& bgr, const cv::Rect& roi) {
    if (bgr.empty()) return {};

    cv::Rect box = roi & cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (box.width < 16 || box.height < 16) box = cv::Rect(0, 0, bgr.cols, bgr.rows);

    cv::Mat work = bgr(box);
    const int scale = std::max(1, cfg_.scale);
    if (scale > 1) {
        cv::resize(work, work, cv::Size(work.cols / scale, work.rows / scale), 0, 0, cv::INTER_LINEAR);
    }

    std::vector<LightBar> bars = findLightBars(work);

    // 检测在缩放图上完成，这里统一换回原图坐标
    for (auto& b : bars) {
        b.top = b.top * static_cast<float>(scale) + cv::Point2f(static_cast<float>(box.x), static_cast<float>(box.y));
        b.bottom = b.bottom * static_cast<float>(scale) + cv::Point2f(static_cast<float>(box.x), static_cast<float>(box.y));
        b.center = b.center * static_cast<float>(scale) + cv::Point2f(static_cast<float>(box.x), static_cast<float>(box.y));
        b.length *= scale;
        b.width *= scale;
        b.area *= scale * scale;
    }

    std::vector<Armor> armors = pairArmors(bars);
    if (armors.empty() && box != cv::Rect(0, 0, bgr.cols, bgr.rows)) {
        // ROI 内丢失目标时回退到全图，避免跟踪框偏了之后一直找不到
        return detect(bgr, cv::Rect());
    }
    return armors;
}
