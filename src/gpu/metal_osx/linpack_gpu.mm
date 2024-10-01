#import "linpack_gpu.h"
#include <iostream>
#include <algorithm>
#include <cmath>

LinpackGPU::LinpackGPU(int matrix_size, int repetitions)
    : n(matrix_size), repetitions(repetitions), a(n * n), b(n) {
    std::srand(std::time(nullptr));  // Инициализация генератора случайных чисел
}

void LinpackGPU::runTest(double& elapsed_time, double& norma) {
    norma = 0.0;

    // Засекаем время работы
    auto start = std::chrono::high_resolution_clock::now();

    for (int rep = 0; rep < repetitions; ++rep) {
        // Генерация матриц
        matgen(norma);

        // Выполнение вычислений с использованием Metal
        metalCompute();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    elapsed_time = elapsed.count();
}

void LinpackGPU::matgen(double& norma) {
    norma = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = static_cast<double>(rand()) / RAND_MAX;
            norma = std::max(norma, std::fabs(a[i * n + j]));
        }
        b[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}


void LinpackGPU::metalCompute() {
    // Получение устройства Metal
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Error: This system does not support Metal!" << std::endl;
        return;
    }

    // Создание команды очереди
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    // Загрузка шейдера из файла
    NSError *error = nil;
    NSString *shaderPath = [[NSBundle mainBundle] pathForResource:@"gpu" ofType:@"metal"];
    NSString *shaderSource = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&error];
    if (!shaderSource) {
        std::cerr << "Error: Could not read Metal shader file." << std::endl;
        return;
    }

    // Создание библиотеки из загруженного шейдера
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library) {
        std::cerr << "Error: Failed to create library from shader source. " << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }

    // Создание MTLFunction и PipelineState
    id<MTLFunction> function = [library newFunctionWithName:@"matrix_multiply"];
    if (!function) {
        std::cerr << "Error: Failed to find function 'matrix_multiply' in the library." << std::endl;
        return;
    }

    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipelineState) {
        std::cerr << "Error: Failed to create compute pipeline state. " << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }

    // Создание буферов данных
    id<MTLBuffer> bufferA = [device newBufferWithBytes:a.data()
                                                length:sizeof(double) * a.size()
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithBytes:b.data()
                                                length:sizeof(double) * b.size()
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = [device newBufferWithLength:sizeof(double) * a.size()
                                                options:MTLResourceStorageModeShared];

    // Создание командного буфера и выполнение команды
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:pipelineState];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferC offset:0 atIndex:2];

    // Определение сетки потоков
    MTLSize gridSize = MTLSizeMake(a.size(), 1, 1);
    NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > a.size()) {
        threadGroupSize = a.size();
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];

    // Коммитим и ждем завершения
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Копирование данных обратно в CPU
    double *results = static_cast<double *>(bufferC.contents);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = results[i];
    }
}
