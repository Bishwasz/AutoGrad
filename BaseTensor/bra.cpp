#include <cblas.h>
#include <iostream>

int main() {
    const int M = 2, N = 2, K = 2;
    float A[M*K] = {
        1, 2, 3, 4,
        
    }; // 2x4
    float B[K*N] = {5, 6 ,
                    7, 8
                   }; // 2x2
    float C[M*N] = {
        0, 0,
        0, 0
    }; // 2x2

    float alpha = 1.0f, beta = 0.0f;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha,
                A, K, B, N, beta,
                C, N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i*N + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
