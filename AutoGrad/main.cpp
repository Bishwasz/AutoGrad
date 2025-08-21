#include "../AutoGrad/AutoGradTensor.h"
#include "../Loss/loss.hpp"
#include <vector>
#include <iostream>
#include <random>
#include <stdexcept>
#include <chrono>


void print_shape(const std::vector<int>& shape) {
    std::cout << "[";
    for (size_t i = 0; i < static_cast<size_t>(shape.size()); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

void print_data(const std::vector<float>& data) {
    std::cout << "[";
    for (size_t i = 0; i < static_cast<size_t>(data.size()); ++i) {
        std::cout << data[i];
        if (i < data.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {
    using namespace std::chrono;



    // AutogradTensor<float> tensor1({2, 2}, {1, 2, 3, 4}, true);
    // AutogradTensor<float> tensor2({2, 2}, {5, 6, 7, 8}, true);

    // Start timer
    auto start = high_resolution_clock::now();
    AutogradTensor<float> tensor3({3,3},{1,2,3,4,5,6,7,8,9},true);
    AutogradTensor<float> tensor4({3}, {1,2,3}, true);
    auto result = tensor3.broadcast_add(tensor4);
    result.backward();
    print_data(result.grad().data());
    print_data(tensor4.grad().data());
    print_data(tensor3.grad().data());
    print_data(result.data());

    // print_data(tensor3.data());

    // Perform multiplication
    // auto result = tensor1 * tensor2;
    // result.backward();
    // // print_data(result.grad().data());
    // print_data(tensor1.grad().data());
    // print_data(tensor2.grad().data());
    // tensor1 -= tensor1.grad() * 0.1f;
    // tensor2 -= tensor2.grad() * 0.1f;

    // print_data(tensor1.data());
    // print_data(tensor2.data());

    // End timer
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    // Print result and timing
    // print_data(result.data());
    std::cout << "Multiplication took: " << duration.count() << " microseconds" << std::endl;

    return 0;
}