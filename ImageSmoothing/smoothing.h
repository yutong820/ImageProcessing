#pragma once
#include <opencv2/opencv.hpp>
#include <string>

cv::Mat applySmoothing(const cv::Mat& input, const std::string& method, int param1, int param2 = 0);