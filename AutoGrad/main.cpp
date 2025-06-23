#include "../AutoGrad/AutoGradTensor.h"
#include "../Loss/loss.hpp"
#include <vector>
#include <iostream>
#include <random>
#include <stdexcept>

void print_shape(const std::vector<int>& shape) {
    std::cout << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

void print_data(const std::vector<float>& data) {
    std::cout << "[";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i];
        if (i < data.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {
    // Initialize predictions (logits) with shape [2, 3] (batch_size=2, num_classes=3)
    AutogradTensor<float> predictions({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);
    
    // Initialize one-hot encoded labels with shape [2, 3]
    AutogradTensor<float> labels({2, 3}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, false);

    // Print inputs


    // Compute cross-entropy loss using Loss class
    auto loss = Loss<float>::cross_entropy_loss(predictions, labels);
    std::cout << "Cross-entropy loss: " << loss.at({0}) << "\n";

    // // Perform backpropagation
    loss.backward();
    // // std::cout << "Loss gradient: ";
    // print_data(loss.grad());
    // std::cout << "Predictions gradient: ";
    print_data(predictions.grad());


    // // Print gradients
    // std::cout << "Predictions gradient: ";
    // print_data(predictions.grad());

    return 0;
}