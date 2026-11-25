#include "aimer.hpp"
#include "calibration.hpp"
#include "detector.hpp"
#include "digit_recognizer.hpp"
#include "tracker.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

template <typename T>
void readNode(const YAML::Node& node, const char* key, T& out) {
    if (node && node[key]) out = node[key].as<T>();
}

AppConfig loadConfig(const std::string& path) {
    AppConfig cfg;
    YAML::Node root = YAML::LoadFile(path);

    readNode(root, "dataset", cfg.dataset);
    readNode(root, "output", cfg.output);
    readNode(root, "model", cfg.model);
    readNode(root, "out_width", cfg.out_width);
    readNode(root, "out_height", cfg.out_height);
    readNode(root, "out_fps", cfg.out_fps);
    readNode(root, "stride", cfg.stride);
    readNode(root, "max_frames", cfg.max_frames);
    readNode(root, "benchmark", cfg.benchmark);
    readNode(root, "draw", cfg.draw);
    readNode(root, "use_digit", cfg.use_digit);

    if (root["detector"]) {
        const auto n = root["detector"];
        std::string color;
        readNode(n, "enemy_color", color);
        if (color == "blue" || color == "BLUE") cfg.detector.enemy_color = 1;
        readNode(n, "min_saturation", cfg.detector.min_saturation);
        readNode(n, "min_value", cfg.detector.min_value);
        readNode(n, "red_hue_max", cfg.detector.red_hue_max);
        readNode(n, "blue_hue_min", cfg.detector.blue_hue_min);
        readNode(n, "blue_hue_max", cfg.detector.blue_hue_max);
        readNode(n, "morph_size", cfg.detector.morph_size);
        readNode(n, "min_light_area", cfg.detector.min_light_area);
        readNode(n, "min_light_length", cfg.detector.min_light_length);
        readNode(n, "min_light_ratio", cfg.detector.min_light_ratio);
        readNode(n, "max_light_ratio", cfg.detector.max_light_ratio);
        readNode(n, "max_tilt", cfg.detector.max_tilt);
        readNode(n, "min_pair_ratio", cfg.detector.min_pair_ratio);
        readNode(n, "max_pair_ratio", cfg.detector.max_pair_ratio);
        readNode(n, "max_angle_diff", cfg.detector.max_angle_diff);
        readNode(n, "max_y_diff_ratio", cfg.detector.max_y_diff_ratio);
        readNode(n, "min_overlap_ratio", cfg.detector.min_overlap_ratio);
        readNode(n, "min_length_ratio", cfg.detector.min_length_ratio);
        readNode(n, "min_armor_width", cfg.detector.min_armor_width);
        readNode(n, "min_distance", cfg.detector.min_distance);
        readNode(n, "max_distance", cfg.detector.max_distance);
        readNode(n, "large_ratio_thresh", cfg.detector.large_ratio_thresh);
        readNode(n, "max_reproj", cfg.detector.max_reproj);
        readNode(n, "max_armors", cfg.detector.max_armors);
        readNode(n, "scale", cfg.detector.scale);
        readNode(n, "use_roi", cfg.detector.use_roi);
        readNode(n, "roi_padding", cfg.detector.roi_padding);
    }

    if (root["tracker"]) {
        const auto n = root["tracker"];
        readNode(n, "min_hits", cfg.tracker.min_hits);
        readNode(n, "max_misses", cfg.tracker.max_misses);
        readNode(n, "assoc_dist_ratio", cfg.tracker.assoc_dist_ratio);
        readNode(n, "velocity_smooth", cfg.tracker.velocity_smooth);
        readNode(n, "digit_vote_min", cfg.tracker.digit_vote_min);
        readNode(n, "digit_conf_thresh", cfg.tracker.digit_conf_thresh);
    }

    if (root["aimer"]) {
        const auto n = root["aimer"];
        readNode(n, "enable", cfg.aimer.enable);
        readNode(n, "bullet_speed", cfg.aimer.bullet_speed);
        readNode(n, "gravity", cfg.aimer.gravity);
        readNode(n, "yaw_offset", cfg.aimer.yaw_offset);
        readNode(n, "pitch_offset", cfg.aimer.pitch_offset);
        readNode(n, "max_gimbal_speed", cfg.aimer.max_gimbal_speed);
        readNode(n, "fire_yaw_tol", cfg.aimer.fire_yaw_tol);
        readNode(n, "fire_pitch_tol", cfg.aimer.fire_pitch_tol);
        readNode(n, "fire_frames", cfg.aimer.fire_frames);
        readNode(n, "prediction_time", cfg.aimer.prediction_time);
    }
    return cfg;
}

void printUsage() {
    std::cout
        << "usage: rm_autoaim [options]\n"
        << "  --config <path>      配置文件，默认 configs/demo.yaml\n"
        << "  --dataset <path>     数据目录，覆盖配置文件\n"
        << "  --out <path>         输出视频\n"
        << "  --stride <n>         每 n 帧处理一帧\n"
        << "  --max-frames <n>     最多处理 n 帧\n"
        << "  --fast               开启跟踪 ROI 缩小检测范围\n"
        << "  --benchmark          只跑耗时统计，不写视频\n"
        << "  --no-digit           关闭数字识别\n"
        << "  --no-draw            不绘制结果\n";
}

void drawArmor(cv::Mat& img, const Armor& a) {
    if (a.corners.size() != 4) return;
    const cv::Scalar color = (a.type == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 200, 255);
    std::vector<cv::Point> pts;
    for (const auto& p : a.corners) pts.emplace_back(cvRound(p.x), cvRound(p.y));
    cv::polylines(img, pts, true, color, 2, cv::LINE_AA);
    cv::circle(img, cv::Point(cvRound(a.center.x), cvRound(a.center.y)), 4,
               cv::Scalar(0, 255, 255), -1, cv::LINE_AA);

    const std::string label = cv::format("%s %.2f", a.type == 0 ? "S" : "L", a.ratio);
    cv::putText(img, label, cv::Point(cvRound(a.center.x) + 8, cvRound(a.center.y) - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
}

void drawTrack(cv::Mat& img, const Track& t, bool confirmed) {
    const cv::Scalar color = confirmed ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 180, 255);
    cv::Rect box(cvRound(t.bbox.x), cvRound(t.bbox.y), cvRound(t.bbox.width),
                 cvRound(t.bbox.height));
    box &= cv::Rect(0, 0, img.cols, img.rows);
    if (box.area() > 0) cv::rectangle(img, box, color, 1, cv::LINE_AA);

    const cv::Point c(cvRound(t.center.x), cvRound(t.center.y));
    cv::circle(img, c, 2, color, -1, cv::LINE_AA);

    std::string label = cv::format("#%d", t.id);
    if (t.digit >= 0) label += cv::format(" D%d", t.digit);
    cv::putText(img, label, cv::Point(box.x, std::max(14, box.y - 6)), cv::FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv::LINE_AA);
}

void drawHud(cv::Mat& img, const std::vector<std::string>& lines, bool fire) {
    const int line_h = 22;
    const int width = 560;
    const int height = static_cast<int>(lines.size()) * line_h + 14;
    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, cv::Rect(10, 10, width, height), cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.68, img, 0.32, 0.0, img);

    int y = 30;
    for (const auto& line : lines) {
        cv::putText(img, line, cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                    cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
        y += line_h;
    }
    if (fire) {
        cv::putText(img, "FIRE", cv::Point(width - 70, 34), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
}

struct Stats {
    std::vector<double> total;
    std::vector<double> detect;
    std::vector<double> digit;
    int frames = 0;
    int detections = 0;
    int hit_frames = 0;
    int digit3 = 0;
    int fire = 0;
};

}  // namespace

int main(int argc, char** argv) {
    std::string config_path = "configs/demo.yaml";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    AppConfig cfg;
    try {
        cfg = loadConfig(config_path);
    } catch (const std::exception& e) {
        std::cerr << "read config failed: " << e.what() << "\n";
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            ++i;
        } else if (arg == "--dataset" && i + 1 < argc) {
            cfg.dataset = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            cfg.output = argv[++i];
        } else if (arg == "--stride" && i + 1 < argc) {
            cfg.stride = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--max-frames" && i + 1 < argc) {
            cfg.max_frames = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--fast") {
            cfg.detector.use_roi = true;
            cfg.fast = true;
        } else if (arg == "--benchmark") {
            cfg.benchmark = true;
            cfg.draw = false;
        } else if (arg == "--no-digit") {
            cfg.use_digit = false;
        } else if (arg == "--no-draw") {
            cfg.draw = false;
        }
    }

    if (cfg.dataset.empty()) {
        std::cerr << "dataset is empty\n";
        return 1;
    }

    const std::string video_path = cfg.dataset + "/video/video.avi";
    const std::string calib_path = cfg.dataset + "/calibration/calibration.yaml";
    const std::string frames_csv = cfg.dataset + "/video/frames.csv";
    const std::string imu_csv = cfg.dataset + "/imu/frame_imu.csv";

    Calibration calib;
    if (!calib.load(calib_path)) {
        std::cerr << "cannot load " << calib_path << "\n";
        return 1;
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "cannot open " << video_path << "\n";
        return 1;
    }

    const double src_fps = cap.get(cv::CAP_PROP_FPS);
    const int src_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const std::vector<double> timestamps = loadTimestamps(frames_csv);
    const std::vector<cv::Vec4d> quaternions = loadQuaternions(imu_csv);

    std::cout << "video: " << video_path << "\n";
    std::cout << "frames: " << src_frames << ", fps: " << src_fps
              << ", timestamps: " << timestamps.size() << ", imu: " << quaternions.size() << "\n";

    Detector detector(cfg.detector, calib);
    DigitRecognizer digit;
    const bool digit_ready = cfg.use_digit && digit.load(cfg.model);
    std::cerr << "digit model: " << (digit_ready ? "ok" : "disabled") << "\n";
    Tracker tracker(cfg.tracker);
    Aimer aimer(cfg.aimer);

    int out_w = cfg.out_width > 0 ? cfg.out_width : static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int out_h = cfg.out_height > 0 ? cfg.out_height : static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    out_w &= ~1;
    out_h &= ~1;

    if (!cfg.benchmark) {
        const std::filesystem::path out_path(cfg.output);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }
    }

    cv::VideoWriter writer;
    if (!cfg.benchmark) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(cfg.output, fourcc, cfg.out_fps, cv::Size(out_w, out_h));
        if (!writer.isOpened()) {
            fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
            writer.open(cfg.output, fourcc, cfg.out_fps, cv::Size(out_w, out_h));
        }
        if (!writer.isOpened()) {
            std::cerr << "cannot open output video " << cfg.output << "\n";
            return 1;
        }
    }

    std::ofstream results;
    if (!cfg.benchmark) {
        results.open("output/results.csv");
        results << "frame,time_s,num_armors,track_id,digit,x,y,z,yaw,pitch,fire,detect_ms,total_ms\n";
    }

    Stats stats;
    int frame_index = 0;
    int processed = 0;
    double last_time = -1.0;
    bool preview_saved = false;

    cv::Mat frame;
    while (cap.read(frame)) {
        const int idx = frame_index++;
        if (idx % cfg.stride != 0) continue;
        if (cfg.max_frames > 0 && processed >= cfg.max_frames) break;

        double t = (idx < static_cast<int>(timestamps.size())) ? timestamps[idx]
                                                               : idx / std::max(1.0, src_fps);
        double dt = (last_time >= 0.0) ? (t - last_time) : (cfg.stride / std::max(1.0, src_fps));
        if (dt < 1e-4 || dt > 0.2) dt = cfg.stride / std::max(1.0, src_fps);
        last_time = t;

        const cv::Vec4d q = (idx < static_cast<int>(quaternions.size())) ? quaternions[idx]
                                                                         : cv::Vec4d(1, 0, 0, 0);
        const cv::Matx33d R_world_gimbal =
            Calibration::quatToMatrix(q[0], q[1], q[2], q[3]) * calib.R_gimbal2imu;

        const double t0 = nowMs();

        cv::Rect roi;
        if (cfg.detector.use_roi) roi = tracker.bestRoi(frame.size(), cfg.detector.roi_padding);
        std::vector<Armor> armors = detector.detect(frame, roi);
        const double t1 = nowMs();

        if (digit_ready) {
            for (auto& a : armors) {
                if (a.type != 0) continue;
                const cv::Mat warp = makeArmorWarp(frame, a);
                int d = -1;
                float conf = 0.f;
                if (digit.recognize(warp, d, conf)) {
                    a.digit = d;
                    a.digit_conf = conf;
                }
            }
        }
        const double t2 = nowMs();

        tracker.update(armors, dt);
        const Track* best = tracker.best();
        const AimState aim = aimer.update(best, dt);
        const double t3 = nowMs();

        cv::Vec3d world{0, 0, 0};
        if (best) world = R_world_gimbal * best->xyz_gimbal;

        if (cfg.draw) {
            for (const auto& a : armors) drawArmor(frame, a);
            for (const auto& tr : tracker.tracks()) {
                drawTrack(frame, tr, tr.confirmed(cfg.tracker.min_hits));
            }

            std::vector<std::string> hud;
            hud.push_back(cv::format("frame %06d   t %.3f s", idx, t));
            hud.push_back(cv::format("detect %.2f ms   digit %.2f ms   total %.2f ms   %.1f FPS",
                                     t1 - t0, t2 - t1, t3 - t0,
                                     1000.0 / std::max(1e-3, t3 - t0)));
            if (best) {
                const char* type = (best->type == 0) ? "small" : "large";
                const std::string digit =
                    (best->digit >= 0) ? std::to_string(best->digit) : std::string("?");
                hud.push_back(cv::format("armor #%d   %s   digit %s   d %.2f m", best->id,
                                         type, digit.c_str(), cv::norm(best->xyz_cam)));
                hud.push_back(cv::format("cam    x %+.3f  y %+.3f  z %+.3f", best->xyz_cam[0],
                                         best->xyz_cam[1], best->xyz_cam[2]));
                hud.push_back(cv::format("gimbal x %+.3f  y %+.3f  z %+.3f", best->xyz_gimbal[0],
                                         best->xyz_gimbal[1], best->xyz_gimbal[2]));
                hud.push_back(cv::format("world  x %+.2f  y %+.2f  z %+.2f", world[0], world[1],
                                         world[2]));
            } else {
                hud.push_back("no target");
            }
            if (aim.valid) {
                hud.push_back(cv::format("aim yaw %+.2f  pitch %+.2f  err %+.2f / %+.2f deg",
                                         aim.target_yaw, aim.target_pitch, aim.yaw_err,
                                         aim.pitch_err));
            } else {
                hud.push_back("aim waiting");
            }
            drawHud(frame, hud, aim.fire);
        }

        const double t4 = nowMs();
        const double total_ms = (cfg.draw ? t4 : t3) - t0;

        if (!cfg.benchmark) {
            cv::Mat out;
            if (frame.cols != out_w || frame.rows != out_h) {
                cv::resize(frame, out, cv::Size(out_w, out_h), 0, 0, cv::INTER_AREA);
            } else {
                out = frame;
            }
            writer.write(out);
            if (!preview_saved && processed >= 120 && best) {
                cv::imwrite("output/preview.jpg", out);
                preview_saved = true;
            }

            if (results.is_open()) {
                results << idx << ',' << std::fixed << std::setprecision(6) << t << ','
                        << armors.size() << ',' << (best ? best->id : -1) << ','
                        << (best ? best->digit : -1) << ',' << std::setprecision(4)
                        << (best ? best->xyz_cam[0] : 0.0) << ','
                        << (best ? best->xyz_cam[1] : 0.0) << ','
                        << (best ? best->xyz_cam[2] : 0.0) << ','
                        << (aim.valid ? aim.target_yaw : 0.0) << ','
                        << (aim.valid ? aim.target_pitch : 0.0) << ','
                        << (aim.fire ? 1 : 0) << ',' << std::setprecision(3) << (t1 - t0) << ','
                        << total_ms << '\n';
            }
        }

        stats.frames++;
        stats.detections += static_cast<int>(armors.size());
        if (!armors.empty()) stats.hit_frames++;
        stats.total.push_back(total_ms);
        stats.detect.push_back(t1 - t0);
        stats.digit.push_back(t2 - t1);
        if (best && best->digit == 3) stats.digit3++;
        if (aim.fire) stats.fire++;
        processed++;

        if (processed % 100 == 0) {
            std::cerr << "\rprocessed " << processed << " frames, last " << std::fixed
                      << std::setprecision(1) << total_ms << " ms" << std::flush;
        }
    }
    std::cerr << "\n";

    if (!cfg.benchmark) {
        writer.release();
        if (results.is_open()) results.close();
    }

    const double avg_total = stats.total.empty() ? 0.0 : [&] {
        double s = 0;
        for (double v : stats.total) s += v;
        return s / stats.total.size();
    }();
    const double avg_detect = stats.detect.empty() ? 0.0 : [&] {
        double s = 0;
        for (double v : stats.detect) s += v;
        return s / stats.detect.size();
    }();

    std::ostringstream summary;
    summary << "frames: " << stats.frames << "\n";
    summary << "detections: " << stats.detections << "\n";
    summary << "coverage: " << std::fixed << std::setprecision(1)
            << (stats.frames > 0 ? 100.0 * stats.hit_frames / stats.frames : 0.0) << " %\n";
    summary << "digit3 frames: " << stats.digit3 << "\n";
    summary << "fire frames: " << stats.fire << "\n";
    summary << std::fixed << std::setprecision(2);
    summary << "detect avg " << avg_detect << " ms, p95 " << percentile(stats.detect, 0.95)
            << " ms\n";
    summary << "total  avg " << avg_total << " ms, p50 " << percentile(stats.total, 0.50)
            << " ms, p95 " << percentile(stats.total, 0.95) << " ms\n";
    summary << "throughput " << (avg_total > 0 ? 1000.0 / avg_total : 0.0) << " FPS\n";

    std::cout << summary.str();
    if (!cfg.benchmark) {
        std::ofstream f("output/summary.txt");
        f << summary.str();
    }
    return 0;
}
