#include "gamma_correction.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

void applyGammaCorrection(const cv::Mat& input, cv::Mat& output, double gamma) {
    CV_Assert(gamma >= 0);
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    cv::LUT(input, lookUpTable, output);
}
