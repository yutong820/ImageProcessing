#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <math.h> 

__global__ void prewittKernel(const unsigned char* input, unsigned char* output, int width, int height) {
    // the image pixel coordinates (x, y) that the current thread is responsible for
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) return;

    // manually implement Prewitt convolution
    int Gx = -input[(y - 1) * width + (x - 1)] - input[y * width + (x - 1)] - input[(y + 1) * width + (x - 1)]
        + input[(y - 1) * width + (x + 1)] + input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)];
                                                         
    int Gy = -input[(y - 1) * width + (x - 1)] - input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)]
        + input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

    // root square and truncate in case overleaf
    int mag = fminf(255, int(sqrtf(Gx * Gx + Gy * Gy)));
    output[y * width + x] = static_cast<unsigned char>(mag);
}

extern "C"
void launchPrewitt(const unsigned char* input, unsigned char* output, int width, int height, cudaStream_t stream) {
    dim3 threads(16, 16); // 16*16 threads in each thread block, each block max is 1024 threads
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y); // cover the whole image
    prewittKernel << <blocks, threads, 0, stream >> > (input, output, width, height);
}
