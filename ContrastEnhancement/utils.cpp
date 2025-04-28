#include "utils.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void drawHistogram(const cv::Mat& img, const std::string& windowName) {
    // convert image into gray image
    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img.clone();

    // caculate histogram
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // normalization
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255,255,255));
    normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX);

    for (int i = 0; i < histSize; i++) {
        cv::rectangle(histImage,
            cv::Point(bin_w * i, hist_h),
            cv::Point(bin_w * (i + 1), hist_h - cvRound(hist.at<float>(i))),
            cv::Scalar(0, 0, 0),
            cv::FILLED);
    }
    line(histImage, cv::Point(25, 25), cv::Point(25, hist_h + 25), cv::Scalar(0, 0, 0), 2); // y axis
    line(histImage, cv::Point(25, hist_h + 25), cv::Point(hist_w + 25, hist_h + 25), cv::Scalar(0, 0, 0), 2); // x axis

    putText(histImage, "0", cv::Point(20, hist_h + 45), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    putText(histImage, "255", cv::Point(hist_w - 20, hist_h + 45), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    putText(histImage, "Pixel Count", cv::Point(30, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    cv::imshow(windowName, histImage);
}


void showImage(const cv::Mat& img, const std::string& windowName, int cropWidth, int cropHeight) {
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    int imgWidth = img.cols;
    int imgHeight = img.rows;

    int centerX = imgWidth / 2;
    int centerY = imgHeight / 2;

    int startX = std::max(0, centerX - cropWidth / 2);
    int startY = std::max(0, centerY - cropHeight / 2);

    if (startX + cropWidth > imgWidth) {
        startX = imgWidth - cropWidth;
    }
    if (startY + cropHeight > imgHeight) {
        startY = imgHeight - cropHeight;
    }

    startX = std::max(0, startX);
    startY = std::max(0, startY);

    cv::Rect roi(startX, startY, std::min(cropWidth, imgWidth - startX), std::min(cropHeight, imgHeight - startY));
    cv::Mat cropped = img(roi).clone();

    cv::imshow(windowName, cropped);
}