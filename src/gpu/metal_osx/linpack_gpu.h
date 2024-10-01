#ifndef LINPACK_GPU_H
#define LINPACK_GPU_H

#include <vector>
#include <chrono>

// Платформозависимая часть
#ifdef __OBJC__
#include <Metal/Metal.h>  // Используем правильный заголовок Metal только для Objective-C++
#endif

class LinpackGPU {
public:
    LinpackGPU(int matrix_size, int repetitions);
    void runTest(double& elapsed_time, double& norma);

private:
    int n;
    int repetitions;
    std::vector<double> a, b;
    
    void matgen(double& norma);
    void metalCompute();
};

#endif // LINPACK_GPU_H