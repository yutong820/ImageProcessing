#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif

void launchPrewitt(const unsigned char* input, unsigned char* output, int width, int height, cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif