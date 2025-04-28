#include "resize.h"

// resize image function
void resizeImage(const cv::Mat& input, cv::Mat& output, double fx, double fy) {
    // INTER_LINEAR is default interpolation method£¬zoom in or out of an image
    cv::resize(input, output, cv::Size(), fx, fy, cv::INTER_LINEAR);
}
