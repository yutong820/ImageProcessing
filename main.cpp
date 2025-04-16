#include <opencv2/opencv.hpp>
#include <iostream>
#include "Resize/resize.h"  

int main() {
    // read image
    cv::Mat img = cv::imread("D:\\360MoveData\\Users\\DELL\\Desktop\\opencv\\DSC_0952.JPG");
    if (img.empty()) {
        std::cerr << "Image load failed!" << std::endl;
        return -1;
    }

    //  resizeImage() zoom in 
    cv::Mat resized;
    resizeImage(img, resized, 0.25, 0.25);  // fx = fy = 0.5 reduced to 1/4

    // display
    cv::imshow("origianl", img);
    cv::imshow("resized", resized);

    cv::waitKey(0);\
    return 0;
}
