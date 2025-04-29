#include <opencv2/opencv.hpp>
#include <iostream>
#include "GammaCorrection/gamma_correction.h" 
#include "utils.h"
#include "Resize/resize.h" 

int main() {
    cv::Mat img = cv::imread("D:\\360MoveData\\Users\\DELL\\Desktop\\example.jpg");
    if (img.empty()) {
        std::cerr << "Image load failed!" << std::endl;
        return -1;
    }

    int choice;
    std::cout << "choose work£º\n";
    std::cout << "1. gamma correction\n";
    std::cin >> choice;

    if (choice == 1) {
        double gamma;
        std::cout << "input gamma value: ";
        std::cin >> gamma;

        cv::Mat gammaCorrected;
        cv::Mat original;
        cv::Mat result;
        applyGammaCorrection(img, gammaCorrected, gamma);

        resizeImage(img, original, 0.5, 0.5);
        cv::imshow("original image", original);
        drawHistogram(img, "original histogram");

        resizeImage(gammaCorrected, result, 0.5, 0.5);
        cv::imshow("Corrected image", result);
        drawHistogram(result, "corrected histogram");

        cv::waitKey(0);
    }

    return 0;
}

