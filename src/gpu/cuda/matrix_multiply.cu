// matrix_multiply.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrix_multiply(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float result = 0.0f;
        for (int k = 0; k < n; ++k) {
            result += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = result;
    }
}
