#include "linpack_gpu.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include "matrix_multiply.cuh"  // Include the CUDA kernel header

LinpackGPU::LinpackGPU(int matrix_size, int repetitions)
    : n(matrix_size), repetitions(repetitions), a(n * n), b(n * n), c(n * n) {
    std::srand(std::time(nullptr)); // Random number generator initialization
}

void LinpackGPU::runTest(double &elapsed_time, double &norma) {
    norma = 0.0;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    for (int rep = 0; rep < repetitions; ++rep) {
        // Generate matrices
        matgen(norma);

        // Perform GPU computation using CUDA
        cudaCompute();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    elapsed_time = elapsed.count();
}

void LinpackGPU::matgen(double &norma) {
    norma = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = static_cast<float>(rand()) / RAND_MAX;
            norma = std::max(norma, std::fabs(a[i * n + j]));
        }
        b[i * n + j] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void LinpackGPU::cudaCompute() {
    float *d_a, *d_b, *d_c;
    size_t size = n * n * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    // Configure the grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the CUDA kernel
    matrix_multiply<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
