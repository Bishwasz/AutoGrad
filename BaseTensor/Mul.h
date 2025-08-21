#pragma once
#include <vector>

// Naive matrix multiply
template <typename T>
void matrixMultiply(const std::vector<T>& A, const std::vector<T>& B,
                    std::vector<T>& C, size_t M, size_t K, size_t N);

// OpenBLAS matrix multiply
template <typename T>
void openBlasMultiply(const std::vector<T>& A, const std::vector<T>& B,
                      std::vector<T>& C, size_t M, size_t K, size_t N);
