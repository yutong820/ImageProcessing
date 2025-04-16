#pragma once
#include <opencv2/opencv.hpp>

// 缩放图像函数声明
void resizeImage(const cv::Mat& input, cv::Mat& output, double fx, double fy); //传入原图传出缩放后的图像，以及横向和纵向缩放比例因子
