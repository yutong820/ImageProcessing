#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "smoothing.h"

void putLabel(cv::Mat& img, const std::string& text) {
    cv::putText(img, text, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 255), 1);
}

int main() {
    cv::Mat img = cv::imread("D:/360MoveData/Users/DELL/Desktop/opencv/example.jpg");

    cv::resize(img, img, cv::Size(), 0.2, 0.2);
    std::vector<std::string> methods = { "mean", "gaussian", "median", "bilateral" };
    std::vector<int> params = { 3, 5, 7, 9 };

    std::vector<cv::Mat> all_rows;
    for (const auto& method : methods) {
        std::vector<cv::Mat> row;
        cv::Mat original_copy;
        img.copyTo(original_copy);
        putLabel(original_copy, "Original");
        row.push_back(original_copy);

        for (int param : params) {
            cv::Mat result;

            if (method == "bilateral") {
                result = applySmoothing(img, method, param, 50);
            }
            else if (method == "gaussian") {
                result = applySmoothing(img, method, param, 2);  
            }
            else {
                result = applySmoothing(img, method, param);
            }

            std::string label = method + " (k=" + std::to_string(param) + ")";
            putLabel(result, label);
            row.push_back(result);
        }

        cv::Mat row_img;
        cv::hconcat(row, row_img);
        all_rows.push_back(row_img);
    }

    cv::Mat final_img;
    cv::vconcat(all_rows, final_img);

    cv::namedWindow("Filter Comparison", cv::WINDOW_NORMAL);
    cv::imshow("Filter Comparison", final_img);
    cv::waitKey(0);
    return 0;
}
