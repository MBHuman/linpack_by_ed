#ifndef LINPACK_H
#define LINPACK_H

#include <vector>
#include <cmath>

class Linpack
{
public:
    Linpack(int matrix_size, int repetitions);
    void runTest(double& elapsed_time, double& norma);

private:
    int n;                  // Размер матрицы
    int repetitions;        // Количество повторений
    std::vector<double> a;  // Матрица
    std::vector<double> b;  // Вектор B
    std::vector<int> ipiv;  // Массив для перестановок

    void matgen(double& norma);
    void hpl_dgesv();       // Аналог HPL для решения системы уравнений
    void lu_decomposition();
    void forward_substitution();
    void backward_substitution();
};

#endif // LINPACK_H
