#include "linear.h"

void linearContrastEnhancement(const cv::Mat& input, cv::Mat& output, double alpha, double beta) {
    // linear transform 
    input.convertTo(output, -1, alpha, beta);
}
