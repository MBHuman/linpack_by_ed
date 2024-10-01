#include "linpack.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <chrono>
#ifdef __ARM_NEON__
#include <arm_neon.h>  // NEON SIMD библиотека
#endif
#ifdef _OPENMP
#include <omp.h>  // OpenMP
#endif

// Include Accelerate only if the build is for macOS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>  // For Linux or Windows, use a generic BLAS/LAPACK library like OpenBLAS or Netlib LAPACK
#endif

#include <iostream>

Linpack::Linpack(int matrix_size, int repetitions)
    : n(matrix_size), repetitions(repetitions), a(n * n), b(n), ipiv(n)
{
    std::srand(std::time(nullptr));  // Инициализация генератора случайных чисел
}

void Linpack::runTest(double& elapsed_time, double& norma)
{
    norma = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:norma)  // Parallelizing repetitions to take advantage of multiple threads
    for (int rep = 0; rep < repetitions; ++rep) {
        double local_norma = 0.0;
        matgen(local_norma);
        #pragma omp critical
        {
            norma = std::max(norma, local_norma);
        }
        hpl_dgesv();  // Решение системы уравнений методом LU-разложения
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    elapsed_time = elapsed.count();
}

void Linpack::matgen(double& norma)
{
    norma = 0.0;
    const int block_size = 32768;  // Размер блока для параллельной генерации чисел

    #pragma omp parallel for schedule(static) reduction(max:norma)
    for (int i = 0; i < n; i += block_size) {
        int limit = std::min(i + block_size, n);
        for (int j = i; j < limit; ++j) {
            float local_norma = 0.0f;

            #ifdef __ARM_NEON__
            int k = 0;
            for (; k <= n - 128; k += 128) {
                // Генерация 128 случайных значений одновременно с использованием NEON
                float32x4_t rand_vecs[32];
                for (int vec_idx = 0; vec_idx < 32; ++vec_idx) {
                    rand_vecs[vec_idx] = vdupq_n_f32(static_cast<float>(rand()) / RAND_MAX);
                }

                // Записываем значения в массив
                for (int vec_idx = 0; vec_idx < 32; ++vec_idx) {
                    vst1q_f32(reinterpret_cast<float*>(&a[j * n + k + vec_idx * 4]), rand_vecs[vec_idx]);
                }

                // Вычисляем максимальные значения
                for (int vec_idx = 0; vec_idx < 32; ++vec_idx) {
                    float32x4_t abs_vec = vabsq_f32(rand_vecs[vec_idx]);
                    local_norma = std::max(local_norma, vgetq_lane_f32(abs_vec, 0));
                    local_norma = std::max(local_norma, vgetq_lane_f32(abs_vec, 1));
                    local_norma = std::max(local_norma, vgetq_lane_f32(abs_vec, 2));
                    local_norma = std::max(local_norma, vgetq_lane_f32(abs_vec, 3));
                }
            }
            #endif

            // Завершаем оставшиеся элементы, если они есть
            for (int k = (n / 128) * 128; k < n; ++k) {
                a[j * n + k] = static_cast<double>(rand()) / RAND_MAX;
                local_norma = std::max(local_norma, static_cast<float>(fabs(a[j * n + k])));
            }

            b[j] = static_cast<double>(rand()) / RAND_MAX;

            // Обновляем значение нормы с использованием OpenMP reduction
            norma = std::max(norma, static_cast<double>(local_norma));
        }
    }
}

// Использование LAPACK для ускорения LU-разложения и решения линейной системы
void Linpack::hpl_dgesv() {
    int lda = n; // leading dimension of matrix A
    int info;    // output: info about success or failure

    #pragma omp parallel sections  // Parallelizing the LAPACK calls, assuming they're thread-safe (depends on LAPACK implementation)
    {
        #pragma omp section
        {
            #ifdef __APPLE__
            // Using Accelerate (macOS)
            dgetrf_(&n, &n, a.data(), &lda, ipiv.data(), &info);
            #else
            // Using generic LAPACK (Linux or Windows, e.g., OpenBLAS or Netlib)
            LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, a.data(), lda, ipiv.data());
            #endif

            if (info != 0) {
                #pragma omp critical
                {
                    std::cerr << "LU Decomposition failed, info: " << info << std::endl;
                }
                return;
            }
        }

        #pragma omp section
        {
            #ifdef __APPLE__
            // Использование dgetrs для решения уравнения Ax = b на macOS
            char trans = 'N';
            int nrhs = 1; // Количество правых частей (колонок вектора b)
            dgetrs_(&trans, &n, &nrhs, a.data(), &lda, ipiv.data(), b.data(), &lda, &info);
            #else
            // Использование LAPACKE_dgetrs для решения уравнения Ax = b на других платформах
            char trans = 'N';
            int nrhs = 1;
            LAPACKE_dgetrs(LAPACK_ROW_MAJOR, trans, n, nrhs, a.data(), lda, ipiv.data(), b.data(), lda);
            #endif

            if (info != 0) {
                #pragma omp critical
                {
                    std::cerr << "Solving linear equations failed, info: " << info << std::endl;
                }
            }
        }
    }
}
