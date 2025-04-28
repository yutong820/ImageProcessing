#include <opencv2/opencv.hpp>
#include <iostream>
#include "Histogram/linear.h"   

int main() {
    cv::Mat img = cv::imread("D:\\360MoveData\\Users\\DELL\\Desktop\\example.jpg");
    if (img.empty()) {
        std::cerr << "Image load failed!" << std::endl;
        return -1;
    }

    double alpha, beta;
    std::cout << "input alpha£¨contrast£©: ";
    std::cin >> alpha;
    std::cout << "input beta£¨brightness£©: ";
    std::cin >> beta;

    cv::Mat enhanced;
    linearContrastEnhancement(img, enhanced, alpha, beta);

    cv::imshow("original", img);
    cv::imshow("enhanced", enhanced);
    cv::waitKey(0);
    return 0;
}
