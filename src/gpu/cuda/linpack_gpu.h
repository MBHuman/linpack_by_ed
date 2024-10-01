#ifndef LINPACK_GPU_H
#define LINPACK_GPU_H

#include <vector>
#include <chrono>

#ifdef defined(__linux__) || defined(_WIN32)
#include <cuda_runtime.h> // CUDA for Linux and Windows
#endif

class LinpackGPU {
public:
    LinpackGPU(int matrix_size, int repetitions);
    void runTest(double &elapsed_time, double &norma);

private:
    int n;
    int repetitions;
    std::vector<float> a, b, c;

    void matgen(double &norma);

    void cudaCompute();
};

#endif // LINPACK_GPU_H
