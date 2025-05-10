#include <opencv2/opencv.hpp>
#include "include/edge_methods.hpp"
#include "include/prewitt_cuda.hpp"  // CUDA header
#include <chrono>


void putLabel(cv::Mat& img, const std::string& text) {
    cv::putText(img, text, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
}

void addLabeledResult(std::vector<cv::Mat>& results, const cv::Mat& gray, const std::string& label) {
    cv::Mat color;
    cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    putLabel(color, label);
    results.push_back(color);
}

cv::Mat mergeMagnitude(const cv::Mat& gx, const cv::Mat& gy) {
    cv::Mat magnitude;
    cv::magnitude(gx, gy, magnitude);
    cv::convertScaleAbs(magnitude, magnitude);
    return magnitude;
}

cv::Mat runPrewittCUDA(const cv::Mat& gray) {
    int width = gray.cols;
    int height = gray.rows;
    size_t img_size = width * height * sizeof(unsigned char);

    unsigned char* d_input, * d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMemcpy(d_input, gray.data, img_size, cudaMemcpyHostToDevice);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] launchPrewitt called..." << std::endl;
    launchPrewitt(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    std::cout << "CUDA Prewitt time: " << elapsed.count() << " sec" << std::endl;

    std::vector<unsigned char> result_data(width * height);
    cudaMemcpy(result_data.data(), d_output, img_size, cudaMemcpyDeviceToHost);

    // release memory
    cudaFree(d_input);
    cudaFree(d_output);

    return cv::Mat(height, width, CV_8UC1, result_data.data()).clone();
}

int main() {
    cv::Mat gray = cv::imread("D:/360MoveData/Users/DELL/Desktop/example.jpg", cv::IMREAD_GRAYSCALE);

    std::cout << "[DEBUG] main started\n";
    cv::Mat result = runPrewittCUDA(gray);
    std::cout << "[DEBUG] returned from runPrewittCUDA\n";

    
    cv::resize(gray, gray, cv::Size(), 0.4,0.4);
    std::vector<cv::Mat> all_rows;
    std::vector<cv::Mat> all_cols;

    // Sobel
    {
        std::vector<cv::Mat> row;
        cv::Mat gx, gy;
        cv::Sobel(gray, gx, CV_64F, 1, 0);
        cv::Sobel(gray, gy, CV_64F, 0, 1);
        //addLabeledResult(row, gray, "Original");
        //addLabeledResult(row, applySobel(gray, 1, 0), "Sobel X");
        //addLabeledResult(row, applySobel(gray, 0, 1), "Sobel Y");
        addLabeledResult(all_cols, mergeMagnitude(gx, gy), "Sobel |G|");
        //cv::Mat disp; cv::hconcat(row, disp); all_rows.push_back(disp);
    }

    // CPU Prewitt
    {
        /*std::vector<cv::Mat> row;*/
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cv::Mat gx = applyPrewitt(gray, 1, 0);
        cv::Mat gy = applyPrewitt(gray, 0, 1);
        cv::Mat mag;
        cv::Mat gx_f, gy_f;
        gx.convertTo(gx_f, CV_32F);
        gy.convertTo(gy_f, CV_32F);
        mag = mergeMagnitude(gx_f, gy_f);
        //addLabeledResult(row, gray, "Original");
        //addLabeledResult(row, gx, "Prewitt X");
        //addLabeledResult(row, gy, "Prewitt Y");
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_time = cpu_end - cpu_start;
        std::cout << "CPU Prewitt time: " << cpu_time.count() << " sec" << std::endl;
        addLabeledResult(all_cols, mag, "Prewitt |G|");
  /*      cv::Mat disp; cv::hconcat(row, disp); all_rows.push_back(disp);*/
    }

    // CUDA Prewitt
    cv::Mat cuda_result = runPrewittCUDA(gray);
    addLabeledResult(all_cols, cuda_result, "CUDA Prewitt");


    // Scharr
    {
        std::vector<cv::Mat> row;
        cv::Mat gx, gy;
        cv::Scharr(gray, gx, CV_64F, 1, 0);
        cv::Scharr(gray, gy, CV_64F, 0, 1);
        //addLabeledResult(row, gray, "Original");
        //addLabeledResult(row, applyScharr(gray, 1, 0), "Scharr X");
        //addLabeledResult(row, applyScharr(gray, 0, 1), "Scharr Y");
        addLabeledResult(all_cols, mergeMagnitude(gx, gy), "Scharr |G|");
   /*     cv::Mat disp; cv::hconcat(row, disp); all_rows.push_back(disp);*/
    }

    // Laplacian
    {
        std::vector<cv::Mat> row;
        //addLabeledResult(row, gray, "Original");
        addLabeledResult(all_cols, applyLaplacian(gray), "Laplacian");
        //cv::Mat blank = cv::Mat::zeros(gray.size(), CV_8UC1);
        //addLabeledResult(row, blank, "");
        //addLabeledResult(row, blank, "");
        //cv::Mat disp; cv::hconcat(row, disp); all_rows.push_back(disp);
    }

    // Canny
    {
        std::vector<cv::Mat> row;
  /*      addLabeledResult(row, gray, "Original");*/

        addLabeledResult(all_cols, applyCanny(gray, 100, 200), "Canny");

        cv::Mat nms;
        cv::Mat edge_manual = applyCannyManual(gray, 100, 200, nms);
        addLabeledResult(all_cols, edge_manual, "Manual Canny");

        //cv::Mat nms_vis;
        //cv::normalize(nms, nms, 0, 255, cv::NORM_MINMAX);
        //nms.convertTo(nms_vis, CV_8U);
        //addLabeledResult(row, nms_vis, "NMS");

   /*     cv::Mat disp; cv::hconcat(row, disp); all_rows.push_back(disp);*/
    }


    //// Manual Sobel
    //{
    //    std::vector<cv::Mat> row;
    //    addLabeledResult(row, gray, "Original");
    //    cv::Mat manual = applySobelManual(gray, 100);
    //    addLabeledResult(row, manual, "Manual Sobel");
    //    cv::Mat blank = cv::Mat::zeros(gray.size(), CV_8UC1);
    //    addLabeledResult(row, blank, "");
    //    addLabeledResult(row, blank, "");
    //    cv::Mat disp; cv::hconcat(row, disp); all_rows.push_back(disp);
    //}


    addLabeledResult(all_cols, gray, "Original");
    // Final display
    cv::Mat final_display;
    cv::hconcat(all_cols, final_display);
    cv::namedWindow("Edge Detection Comparison", cv::WINDOW_NORMAL);
    cv::imshow("Edge Detection Comparison", final_display);
    cv::waitKey(0);
    return 0;
}