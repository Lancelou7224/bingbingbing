#include "digit_recognizer.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

bool DigitRecognizer::load(const std::string& model_path) {
    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
        ready_ = !net_.empty();
    } catch (const cv::Exception& e) {
        std::cerr << "load digit model failed: " << e.what() << "\n";
        ready_ = false;
    }
    return ready_;
}

bool DigitRecognizer::recognize(const cv::Mat& armor_warp, int& digit, float& confidence) {
    digit = -1;
    confidence = 0.f;
    if (!ready_ || armor_warp.empty()) return false;

    cv::Mat gray;
    cv::cvtColor(armor_warp, gray, cv::COLOR_BGR2GRAY);
    cv::Mat clahe;
    cv::createCLAHE(2.0, cv::Size(8, 8))->apply(gray, clahe);

    const int h = clahe.rows, w = clahe.cols;
    const cv::Rect roi(static_cast<int>(w * 0.16f), static_cast<int>(h * 0.05f),
                       static_cast<int>(w * 0.68f), static_cast<int>(h * 0.90f));
    cv::Mat roi_img = clahe(roi & cv::Rect(0, 0, w, h));
    if (roi_img.empty()) return false;

    cv::Mat bin;
    cv::threshold(roi_img, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (cv::mean(bin)[0] > 127.0) cv::bitwise_not(bin, bin);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return false;

    const float roi_area = static_cast<float>(roi_img.rows * roi_img.cols);
    cv::Rect best;
    float best_area = 0.f;
    for (const auto& c : contours) {
        const cv::Rect r = cv::boundingRect(c);
        const float area = static_cast<float>(cv::contourArea(c));
        if (area < roi_area * 0.04f || area > roi_area * 0.80f) continue;
        if (r.width > roi_img.cols * 0.92f) continue;
        if (r.height < roi_img.rows * 0.35f) continue;
        const float aspect = static_cast<float>(r.height) / std::max(1, r.width);
        if (aspect < 0.65f || aspect > 3.6f) continue;
        if (area > best_area) {
            best_area = area;
            best = r;
        }
    }
    if (best.area() <= 0) return false;

    cv::Mat glyph = bin(best);
    const float scale = 20.f / static_cast<float>(std::max(glyph.cols, glyph.rows));
    const int nw = std::max(1, static_cast<int>(std::round(glyph.cols * scale)));
    const int nh = std::max(1, static_cast<int>(std::round(glyph.rows * scale)));
    cv::Mat resized;
    cv::resize(glyph, resized, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);

    cv::Mat canvas = cv::Mat::zeros(28, 28, CV_8U);
    const int ox = (28 - nw) / 2;
    const int oy = (28 - nh) / 2;
    resized.copyTo(canvas(cv::Rect(ox, oy, nw, nh)));

    cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(28, 28),
                                          cv::Scalar(), false, false);
    net_.setInput(blob);
    cv::Mat out = net_.forward();
    if (out.empty()) return false;

    cv::Mat logits = out.reshape(1, 1);
    logits.convertTo(logits, CV_64F);
    double max_logit = 0.0;
    cv::minMaxLoc(logits, nullptr, &max_logit);
    cv::Mat probs;
    cv::exp(logits - max_logit, probs);
    const double sum = cv::sum(probs)[0];
    if (sum <= 0.0) return false;
    probs /= sum;

    cv::Point max_loc;
    cv::minMaxLoc(probs, nullptr, nullptr, nullptr, &max_loc);
    digit = max_loc.x;
    confidence = static_cast<float>(probs.at<double>(0, digit));
    return digit >= 0 && digit <= 9;
}
