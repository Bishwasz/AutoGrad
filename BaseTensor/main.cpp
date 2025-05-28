#include "BaseTensor.h"
#include <iostream>
#include <chrono>  // Add this header for timing

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
        std::vector<int> shape = {2, 2};

        BaseTensor<double> tensor1(shape, std::vector<double>{1,2,3,4}); // Initialize with ones
        // BaseTensor<int> tensor2(std::vector<int>{1, 2},std::vector<int>{5,5}); // Initialize with zeros

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        auto result = tensor1.neg(); // Assuming tensor2 is initialized to zeros

        // End timing
        auto end = std::chrono::high_resolution_clock::now();

        // Compute duration
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Matrix multiplication took " << duration.count() << " microseconds\n";

        print_tensor(tensor1);
        print_tensor(*result);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
