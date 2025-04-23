#include "tensor.h" // Assuming the files are saved as tensor.h and tensor.cpp
#include <iostream>
#include <vector>
#include <activations.h>


void print_tensor(const std::vector<float> c) {

    for (const auto& val : c) {
        std::cout << val << " ";
    }
    std::cout << "\n";
}

int main() {
    // --- Example 1: Addition ---
    std::vector<int> shape = {2, 2};
    std::vector<float> a = {1.0, 2.0, 2.0, 1.0};
    std::vector<float> b = {5.0, 6.0, 7.0, 8.0};
    std::vector<float> c = {9.0, 10.0, 11.0, 12.0};
    std::vector<float> d = {1.0, 1.0 };

    std::shared_ptr<Tensor> V = std::make_shared<Tensor>(std::vector<int>{2, 1}, d,true);
    bool requires_grad = true;  
    std::shared_ptr<Tensor> A = std::make_shared<Tensor>(shape, a, requires_grad);
    std::shared_ptr<Tensor> Z = *A * *V;
    auto sigmoid_result = relu(*Z);
    std::cout << "Z result" << std::endl;

    print_tensor(Z->data());
    print_tensor(sigmoid_result->data());

    // std::shared_ptr<Tensor> B = std::make_shared<Tensor>(shape, b, requires_grad);
    // std::shared_ptr<Tensor> C = std::make_shared<Tensor>(shape, c, requires_grad);
    // std::shared_ptr<Tensor> T = *A * *B;
    // std::shared_ptr<Tensor> D = *T * *C;


    // D->backward();
    // std::cout << "C result" << std::endl;
    // print_tensor(D->data());

    // std::cout << "A grad" << std::endl;
    // print_tensor(A->grad());

    // std::cout << "B grad" << std::endl;
    // print_tensor(B->grad());

    // std::cout << "C grad" << std::endl;
    // print_tensor(C->grad());

    // std::cout << std::endl;


    return 0;
}