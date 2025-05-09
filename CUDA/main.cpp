#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

int main() {
    Mat src = imread("D:\\360MoveData\\Users\\DELL\\Desktop\\example.jpg", IMREAD_GRAYSCALE); // gray image, cuda::threshold only supports single channel image
    if (src.empty()) {
        cerr << "Image not found!" << endl;
        return -1;
    }

    vector<int> cuda_supported = { THRESH_BINARY, THRESH_BINARY_INV };
    vector<int> cpu_supported = { THRESH_BINARY, THRESH_BINARY_INV }; //, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV };

    // ---- CPU total time ----
    double cpu_total = 0.0;
    for (int type : cpu_supported) {
        Mat dst;
        int64 t1 = getTickCount();
        threshold(src, dst, 128.0, 255.0, type);
        int64 t2 = getTickCount();
        cpu_total += (t2 - t1) / getTickFrequency(); // converts unit to second
    }
    cout << "CPU Total Time (2 thresholds): " << cpu_total << " sec" << endl;

    // ---- GPU total time ----
    if (cuda::getCudaEnabledDeviceCount() == 0) {
        cerr << "No CUDA device found!" << endl;
        return -1;
    }

    cuda::setDevice(0);
    cuda::GpuMat gpu_src, gpu_dst;
    gpu_src.upload(src); // host to device

    double gpu_total = 0.0;
    for (int type : cuda_supported) {
        int64 t1 = getTickCount();
        cuda::threshold(gpu_src, gpu_dst, 128.0, 255.0, type);
        int64 t2 = getTickCount();

        Mat result;
        gpu_dst.download(result); 
        gpu_total += (t2 - t1) / getTickFrequency();
    }
    cout << "GPU Total Time (2 thresholds): " << gpu_total << " sec" << endl;
    cout << "Speedup (on 2 thresholds): " << (cpu_total / 5.0 * 2.0) / gpu_total << "x" << endl;

    return 0;
}
