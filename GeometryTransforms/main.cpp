#include <opencv2/opencv.hpp>
#include <iostream>
#include "Resize/resize.h"  
#include "Rotate/rotate.h" 

int main() {
    // read image
    cv::Mat img = cv::imread("D:\\360MoveData\\Users\\DELL\\Desktop\\opencv\\DSC_0952.JPG");
    if (img.empty()) {
        std::cerr << "Image load failed!" << std::endl;
        return -1;
    }

    int choice;
    std::cout << "please choose: \n1. image resize\n2. image rotate\ninput number: ";
    std::cin >> choice;

    cv::Mat result;

    if (choice == 1) {
        resizeImage(img, result, 0.25, 0.25);
        cv::imshow("after scaling", result);
    }
    else if (choice == 2) {
        double angle;
        std::cout << "input rotation angle: ";
        std::cin >> angle;
        rotateImage(img, result, angle);
        cv::imshow("after rotation", result);
    }
    else
    {
        std::cerr << "invalid option" << std::endl;
        return -1;
    }

    // display
    cv::imshow("origianl", img);
    cv::waitKey(0);\
    return 0;
}
