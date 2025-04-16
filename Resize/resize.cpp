#include "resize.h"

// 缩放图像函数实现
void resizeImage(const cv::Mat& input, cv::Mat& output, double fx, double fy) {
    // INTER_LINEAR 是默认的插值方式，适合缩小或放大图像
    cv::resize(input, output, cv::Size(), fx, fy, cv::INTER_LINEAR);
}
