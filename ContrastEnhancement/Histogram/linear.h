#pragma once
#include <opencv2/opencv.hpp>

// linear contrast
void linearContrastEnhancement(const cv::Mat& input, cv::Mat& output, double alpha, double beta);
