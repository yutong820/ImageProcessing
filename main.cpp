#include <opencv2/opencv.hpp>
#include <iostream>
#include "Resize/resize.h"  // 引入 Resize 模块

int main() {
    // 读取图像
    cv::Mat img = cv::imread("D:\\360MoveData\\Users\\DELL\\Desktop\\opencv\\DSC_0952.JPG");
    if (img.empty()) {
        std::cerr << "图像加载失败，请确保 example.jpg 在 Debug 目录下！" << std::endl;
        return -1;
    }

    // 调用 resizeImage() 缩小图像
    cv::Mat resized;
    resizeImage(img, resized, 0.25, 0.25);  // fx = fy = 0.5 表示缩小为原图1/4

    // 显示原图和缩放图
    cv::imshow("原图", img);
    cv::imshow("缩放后", resized);

    cv::waitKey(0);\
    return 0;
}
