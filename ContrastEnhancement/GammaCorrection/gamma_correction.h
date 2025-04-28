#pragma once
#include <opencv2/opencv.hpp>

void applyGammaCorrection(const cv::Mat& input, cv::Mat& output, double gamma);
void plotHistogram(const cv::Mat& img, const std::string& windowName);
