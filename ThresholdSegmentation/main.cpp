#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void putLabel(cv::Mat& image, const std::string& text) {
    cv::putText(image, text, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
        0.8, cv::Scalar(0, 0, 255), 2);
}

void addLabeledResult(std::vector<cv::Mat>& results, const cv::Mat& gray, const std::string& label) {
    cv::Mat color;
    cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    putLabel(color, label);
    results.push_back(color);
}


// fixed threshold segmentation
cv::Mat myFixedThreshold(const cv::Mat& img, int thresh) {
    CV_Assert(img.type() == CV_8UC1);  // must be gray 
    cv::Mat result = img.clone();
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            uchar pixel = img.at<uchar>(i, j);
            result.at<uchar>(i, j) = (pixel > thresh) ? 255 : 0;
        }
    }
    return result;
}

// Otsu threshold segmentation
cv::Mat myOtsuThreshold(const cv::Mat& img) {
    CV_Assert(img.type() == CV_8UC1);
    int hist[256] = { 0 };
    int total = img.rows * img.cols;

    // statistical histogram
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            hist[img.at<uchar>(i, j)]++;
        }
    }

    // overall average gray value
    double total_mean = 0;
    for (int i = 0; i < 256; ++i) {
        total_mean += i * hist[i];
    }

    double max_var = 0;
    int best_thresh = 0;
    double sum_b = 0;
    int w_b = 0;

    // iterate through each t value to calculate intr-class
    for (int t = 0; t < 256; ++t) {
        w_b += hist[t]; // background pixel numbers
        if (w_b == 0) continue;
        int w_f = total - w_b;
        if (w_f == 0) break;

        sum_b += t * hist[t]; // background gray sum
        double m_b = sum_b / w_b; // background mean value
        double m_f = (total_mean - sum_b) / w_f; // foreground mean value
        double between_var = w_b * w_f * (m_b - m_f) * (m_b - m_f); //between class variance

        if (between_var > max_var) {
            max_var = between_var;
            best_thresh = t;
        }
    }

    return myFixedThreshold(img, best_thresh);
}

cv::Mat myAdaptiveThreshold(const cv::Mat& img, int blockSize, int C) {
    CV_Assert(img.type() == CV_8UC1);
    CV_Assert(blockSize % 2 == 1);  // odd number

    int r = blockSize / 2;
    cv::Mat result = img.clone();
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            // Local window range
            int x0 = std::max(i - r, 0);
            int x1 = std::min(i + r, img.rows - 1);
            int y0 = std::max(j - r, 0);
            int y1 = std::min(j + r, img.cols - 1);

            // caculate local mean value
            int sum = 0;
            int count = 0;
            for (int x = x0; x <= x1; ++x) {
                for (int y = y0; y <= y1; ++y) {
                    sum += img.at<uchar>(x, y);
                    count++;
                }
            }
            int local_mean = sum / count;

            // apply threshold value
            result.at<uchar>(i, j) = (img.at<uchar>(i, j) > local_mean - C) ? 255 : 0;
        }
    }
    return result;
}


int main() {
   
    cv::Mat gray = cv::imread("D:/360MoveData/Users/DELL/Desktop/example.jpg", cv::IMREAD_GRAYSCALE);

    cv::resize(gray, gray, cv::Size(), 0.4, 0.4);

    std::vector<cv::Mat> results;

    cv::Mat original;
    cv::cvtColor(gray, original, cv::COLOR_GRAY2BGR);
    addLabeledResult(results, gray, "Original");

    cv::Mat fixed = myFixedThreshold(gray, 100);
    cv::cvtColor(fixed, fixed, cv::COLOR_GRAY2BGR);
    addLabeledResult(results, myFixedThreshold(gray, 100), "Fixed (100)");

    cv::Mat otsu = myOtsuThreshold(gray);
    cv::cvtColor(otsu, otsu, cv::COLOR_GRAY2BGR);
    addLabeledResult(results, myOtsuThreshold(gray), "Otsu");

    cv::Mat adaptive = myAdaptiveThreshold(gray, 11, 2);
    cv::cvtColor(adaptive, adaptive, cv::COLOR_GRAY2BGR);
    addLabeledResult(results, myAdaptiveThreshold(gray, 11, 2), "Adaptive");


    cv::Mat concat_img;
    cv::hconcat(results, concat_img);
    cv::namedWindow("Thresholding Comparison", cv::WINDOW_NORMAL);
    cv::imshow("Thresholding Comparison", concat_img);
    cv::waitKey(0);

    return 0;
}
