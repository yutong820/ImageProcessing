#pragma once
#include <cuda_runtime.h>

void launchPrewitt(const unsigned char* input, unsigned char* output, int width, int height, cudaStream_t stream = 0);
