#include "BaseTensor.h"
#include <iostream>
#include <chrono>  // For timing

template <typename T>
void print_tensor(const BaseTensor<T>& tensor) {
    std::cout << "Shape: ";
    for (int dim : tensor.shape()) {
        std::cout << dim << " ";
    }
    std::cout << "\nData: ";
    for (T val : tensor.data()) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    try {
        std::vector<int> shape = {500, 500};

        BaseTensor<double> tensor1(shape);
        BaseTensor<double> tensor2(shape);

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        
        auto tensor4 = tensor1 * tensor2; // Matrix multiplication

        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "Matrix multiplication took " << duration.count() << " ms.\n";


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
