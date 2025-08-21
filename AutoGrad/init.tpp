#include "AutoGradTensor.h"
template<typename T>
AutogradTensor<T>::AutogradTensor(const std::vector<int>& shape, const std::vector<T>& data, bool requires_grad)
    : BaseTensor<T>(shape,data), requires_grad_(requires_grad) {
    

    grad_ = BaseTensor<T>(shape);  // Always initialize
    grad_.zero_data();
}


template<typename T>
AutogradTensor<T>::AutogradTensor(const std::vector<int>& shape, bool requires_grad)
    : BaseTensor<T>(shape), requires_grad_(requires_grad) {

  
    size_t size = this->BaseTensor<T>::compute_size();
    
    // --- Glorot/Xavier Initialization ---
    // fan_in is the number of input neurons, fan_out is the number of output neurons
    int fan_in = (shape.size() > 1) ? shape[1] : shape[0];
    int fan_out = shape[0];
    
    T limit = std::sqrt(6.0f / (fan_in + fan_out));

    std::random_device rd;
    std::mt19937 gen(rd());
    // The distribution is now centered at 0
    std::uniform_real_distribution<T> dist(-limit, limit); 

    for (size_t i = 0; i < size; ++i) {
        (*this->data_)[i] = dist(gen);
    }
    // --- End of Initialization Change ---

    grad_ = BaseTensor<T>(shape);
    grad_.zero_data();
}


