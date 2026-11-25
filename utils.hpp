#pragma once

#include "types.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

double nowMs();

std::vector<double> loadTimestamps(const std::string& path);
std::vector<cv::Vec4d> loadQuaternions(const std::string& path);

cv::Mat makeArmorWarp(const cv::Mat& frame, const Armor& armor, int width = 270, int height = 112);

double percentile(std::vector<double> values, double p);
