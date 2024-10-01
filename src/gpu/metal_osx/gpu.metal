// linpack_kernel.metal
#include <metal_stdlib>
using namespace metal;

kernel void matrix_multiply(device float* a [[buffer(0)]],
                            device float* b [[buffer(1)]],
                            device float* c [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    int row = gid.x;
    int col = gid.y;

    float result = 0.0;
    for (int k = 0; k < 512; ++k) { // Предполагаемый размер матрицы 512
        result += a[row * 512 + k] * b[k * 512 + col];
    }

    c[row * 512 + col] = result;
}
