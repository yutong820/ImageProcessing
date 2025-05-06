#include "../include/edge_methods.hpp"
#include <opencv2/opencv.hpp>

cv::Mat applySobel(const cv::Mat& gray, int dx, int dy) {
    cv::Mat grad;
    cv::Sobel(gray, grad, CV_64F, dx, dy, 3); // data type is CV_64F, sobel size is 3
    cv::convertScaleAbs(grad, grad);
    return grad;
}

cv::Mat applyPrewitt(const cv::Mat& gray, int dx, int dy) {
    cv::Mat kernel;
    if (dx == 1 && dy == 0) {
        kernel = (cv::Mat_<float>(3, 3) << -1, 0, 1,
            -1, 0, 1,
            -1, 0, 1);
    }
    else if (dx == 0 && dy == 1) {
        kernel = (cv::Mat_<float>(3, 3) << -1, -1, -1,
            0, 0, 0,
            1, 1, 1);
    }
    cv::Mat grad;
    cv::filter2D(gray, grad, CV_32F, kernel);
    cv::convertScaleAbs(grad, grad);
    return grad;
}

cv::Mat applyScharr(const cv::Mat& gray, int dx, int dy) {
    cv::Mat grad;
    cv::Scharr(gray, grad, CV_64F, dx, dy);
    cv::convertScaleAbs(grad, grad);// grayscale display of two convo results
    return grad;
}

cv::Mat applyLaplacian(const cv::Mat& gray) {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::convertScaleAbs(lap, lap);
    return lap;
}

cv::Mat applyCanny(const cv::Mat& gray, int low_thresh, int high_thresh) {
    cv::Mat edge;
    cv::Canny(gray, edge, low_thresh, high_thresh);
    return edge;
}

cv::Mat applySobelManual(const cv::Mat& gray, double threshold) {
    cv::Mat gx = cv::Mat::zeros(gray.size(), CV_32F);
    cv::Mat gy = cv::Mat::zeros(gray.size(), CV_32F);
    cv::Mat result = cv::Mat::zeros(gray.size(), CV_8U);

    float sobel_x[3][3] = { {-1, 0, 1},
                           {-2, 0, 2},
                           {-1, 0, 1} };
    float sobel_y[3][3] = { {-1, -2, -1},
                           { 0,  0,  0},
                           { 1,  2,  1} };

    for (int i = 1; i < gray.rows - 1; ++i) {
        for (int j = 1; j < gray.cols - 1; ++j) {
            float sum_x = 0, sum_y = 0; // convolutional calculation, iterate through 3*3 neighbors of each pixel
            for (int m = -1; m <= 1; ++m) {
                for (int n = -1; n <= 1; ++n) {
                    uchar pixel = gray.at<uchar>(i + m, j + n);
                    sum_x += sobel_x[m + 1][n + 1] * pixel;
                    sum_y += sobel_y[m + 1][n + 1] * pixel;
                }
            }
            float magnitude = std::sqrt(sum_x * sum_x + sum_y * sum_y);
            result.at<uchar>(i, j) = (magnitude > threshold) ? 255 : 0; // pixels with gradient magnitude greater than threshold are marked as white(edge)
        }
    }
    return result;
}

cv::Mat applyCannyManual(const cv::Mat& gray, double low_thresh, double high_thresh, cv::Mat& nms) {
    // Step 1: Gaussian Blur
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);

    // Step 2: Gradient and direction (Sobel)
    cv::Mat gx, gy;
    cv::Sobel(blurred, gx, CV_64F, 1, 0);
    cv::Sobel(blurred, gy, CV_64F, 0, 1);

    cv::Mat magnitude, angle;
    cv::cartToPolar(gx, gy, magnitude, angle, true); // magnitude and angle of each pixel

    // Step 3: Non-maximum suppression (simplified)
    nms = cv::Mat::zeros(magnitude.size(), CV_64F); //empty image
    for (int i = 1; i < magnitude.rows - 1; ++i) {  // iterarte iamge internal pixels, exclude boundary
        for (int j = 1; j < magnitude.cols - 1; ++j) {
            double ang = angle.at<double>(i, j); // read magnitude and direction of current pixel
            double mag = magnitude.at<double>(i, j);
            double q = 0, r = 0;

            // Approximate direction to 0, 45, 90, 135 degrees
            if ((0 <= ang && ang < 22.5) || (157.5 <= ang && ang <= 180)) {
                q = magnitude.at<double>(i, j + 1);
                r = magnitude.at<double>(i, j - 1);
            }
            else if (22.5 <= ang && ang < 67.5) {
                q = magnitude.at<double>(i + 1, j - 1);
                r = magnitude.at<double>(i - 1, j + 1);
            }
            else if (67.5 <= ang && ang < 112.5) {
                q = magnitude.at<double>(i + 1, j);
                r = magnitude.at<double>(i - 1, j);
            }
            else if (112.5 <= ang && ang < 157.5) {
                q = magnitude.at<double>(i - 1, j - 1);
                r = magnitude.at<double>(i + 1, j + 1);
            }

            if (mag >= q && mag >= r) {  // value keeped only if current pixel greater than the other two directions pixels, otherwise is 0
                nms.at<double>(i, j) = mag;
            }
            else {
                nms.at<double>(i, j) = 0;
            }
        }
    }

    // Step 4: Double threshold
    cv::Mat edge = cv::Mat::zeros(nms.size(), CV_8U);
    for (int i = 0; i < nms.rows; ++i) {
        for (int j = 0; j < nms.cols; ++j) {
            double val = nms.at<double>(i, j);
            if (val >= high_thresh) edge.at<uchar>(i, j) = 255;
            else if (val >= low_thresh) edge.at<uchar>(i, j) = 100;  // weak edge, just discard
        }
    }

    // Step 5: Edge tracking by hysteresis (simple version)
    for (int i = 1; i < edge.rows - 1; ++i) {
        for (int j = 1; j < edge.cols - 1; ++j) {
            if (edge.at<uchar>(i, j) == 100) {
                // Check 8-connected neighborhood for strong edge
                bool connected = false;
                for (int m = -1; m <= 1; ++m) { // check if there is strong edges around weak edge sourrounding 
                    for (int n = -1; n <= 1; ++n) {
                        if (edge.at<uchar>(i + m, j + n) == 255) {
                            connected = true;
                        }
                    }
                }
                if (connected) edge.at<uchar>(i, j) = 255; // if there is a strong edge in the sourrounding 8 neighbors -- upgrade to a strong edge
                else edge.at<uchar>(i, j) = 0; // otherwise just discard the weak edge
            }
        }
    }
    return edge;
}
