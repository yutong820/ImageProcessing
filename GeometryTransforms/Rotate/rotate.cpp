#include "rotate.h"

void rotateImage(const cv::Mat& input, cv::Mat& output, double angle) {
    // rotating center, 2.0F is half to get image center point
    cv::Point2f center(input.cols / 2.0F, input.rows / 2.0F);

    // get rotate matrix
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);  // scale=1.0 no scaling

    // affine transform
    cv::warpAffine(input, output, rotationMatrix, input.size());
}
