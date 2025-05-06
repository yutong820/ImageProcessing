#pragma once
#include <opencv2/opencv.hpp>

cv::Mat applySobel(const cv::Mat& gray, int dx, int dy);
cv::Mat applyPrewitt(const cv::Mat& gray, int dx, int dy);
cv::Mat applyScharr(const cv::Mat& gray, int dx, int dy);
cv::Mat applyLaplacian(const cv::Mat& gray);
cv::Mat applyCanny(const cv::Mat& gray, int low_thresh, int high_thresh);
cv::Mat applySobelManual(const cv::Mat& gray, double threshold);
cv::Mat applyCannyManual(const cv::Mat& gray, double low_thresh, double high_thresh, cv::Mat& nms_out);
