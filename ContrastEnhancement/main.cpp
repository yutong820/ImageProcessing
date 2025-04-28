#include <opencv2/opencv.hpp>
#include <iostream>
#include "GammaCorrection/gamma_correction.h" 
#include "utils.h"

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
        applyGammaCorrection(img, gammaCorrected, gamma);

        showImage(img, "original image");
        drawHistogram(img, "original histogram");

        showImage(gammaCorrected, "Corrected image");
        drawHistogram(gammaCorrected, "corrected histogram");

        cv::waitKey(0);
    }

    return 0;
}

