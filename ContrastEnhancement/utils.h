#pragma once
#include <opencv2/opencv.hpp>

// draw histogram
void drawHistogram(const cv::Mat& img, const std::string& windowName);

void showImage(const cv::Mat& img, const std::string& windowName, int cropWidth = 1920, int cropHeight = 1080);
