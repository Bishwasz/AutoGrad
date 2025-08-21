#include <vector>
#include <iostream>
#include <cblas.h>
template <typename T>
void matrixMultiply(const std::vector<T>& A, const std::vector<T>& B,
                    std::vector<T>& C, size_t M, size_t K, size_t N) {
    // Assumes: A is MxK, B is KxN, and C is MxN (preallocated)
    // std::cout << "Performing matrix multiplication: " << M << "x" << K << " * " << K << "x" << N << "\n";
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
template <typename T>
void openBlasMultiply(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, size_t M, size_t K, size_t N) {
    // Placeholder for OpenBLAS implementation
    // std::cout << "Using OpenBLAS for matrix multiplication: " << M << "x" << K << " * " << K << "x" << N << "\n";
    // Actual OpenBLAS call would go here
        const T alpha = static_cast<T>(1.0);
    const T beta = static_cast<T>(0.0);
      if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    alpha,
                    A.data(), static_cast<int>(K),
                    B.data(), static_cast<int>(N),
                    beta,
                    C.data(), static_cast<int>(N));
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    alpha,
                    A.data(), static_cast<int>(K),
                    B.data(), static_cast<int>(N),
                    beta,
                    C.data(), static_cast<int>(N));
    } }

