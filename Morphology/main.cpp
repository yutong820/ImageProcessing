#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void putLabel(cv::Mat& image, const std::string& text) {
    cv::putText(image, text, cv::Point(10, 25),
        cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(0, 0, 255), 2);
}
void addLabeledResult(std::vector<cv::Mat>& results, const cv::Mat& gray, const std::string& label) {
    cv::Mat color;
    cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    putLabel(color, label);
    results.push_back(color);
}

int main() {
    cv::Mat gray = cv::imread("D:/360MoveData/Users/DELL/Desktop/example.jpg", cv::IMREAD_GRAYSCALE);

    cv::resize(gray, gray, cv::Size(), 0.4, 0.4); 

    // binary, Otsu threshold 
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // define structure element
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    cv::Mat eroded, dilated, opened, closed;
    cv::erode(binary, eroded, kernel);
    cv::dilate(binary, dilated, kernel);
    cv::morphologyEx(binary, opened, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary, closed, cv::MORPH_CLOSE, kernel);

    std::vector<cv::Mat> results;
    addLabeledResult(results, binary, "Original (Otsu)");
    addLabeledResult(results, eroded, "Erosion");
    addLabeledResult(results, dilated, "Dilation");
    addLabeledResult(results, opened, "Opening");
    addLabeledResult(results, closed, "Closing");

    cv::Mat concat_img;
    cv::hconcat(results, concat_img);
    cv::namedWindow("Morphological Operations", cv::WINDOW_NORMAL);
    cv::imshow("Morphological Operations", concat_img);
    cv::waitKey(0);

    return 0;
}
