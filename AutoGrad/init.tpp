#include "AutoGradTensor.h"
template<typename T>
AutogradTensor<T>::AutogradTensor(const std::vector<int>& shape, const std::vector<T>& data, bool requires_grad)
    : BaseTensor<T>(shape, data), requires_grad_(requires_grad) {
        size_t size = this->BaseTensor<T>::compute_size();
        if (data.size() != size) {
            std::string shape_str;
            for (size_t i = 0; i < this->shape_.size(); ++i) {
                shape_str += std::to_string(this->shape_[i]) + (i == this->shape_.size() - 1 ? "" : ", ");
            }
            throw std::runtime_error("Data size (" + std::to_string(data.size()) + ") does not match shape [" + shape_str + "] which requires size " + std::to_string(size));
        }
        this->data_ = std::make_unique<std::vector<T>>(data);
        if (requires_grad_) {
            this->grad_ = std::make_unique<std::vector<T>>(size, T{0});
        }
} 
template<typename T>
AutogradTensor<T>::AutogradTensor(const std::vector<int>& shape, bool requires_grad)
    : BaseTensor<T>(shape), requires_grad_(requires_grad) {

    size_t size = this->BaseTensor<T>::compute_size();
    this->data_ = std::make_unique<std::vector<T>>(size);

    std::random_device rd;
    std::mt19937 gen(rd());

    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(0, 100);
        for (size_t i = 0; i < size; ++i) {
            (*this->data_)[i] = dist(gen);
        }
    } else if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(0.0, 1.0);
        for (size_t i = 0; i < size; ++i) {
            (*this->data_)[i] = dist(gen);
        }
    } else {
        throw std::runtime_error("Random initialization not supported for this type");
    }

    // ✅ ALWAYS initialize grad_ if required
    if (requires_grad_) {
        this->grad_ = std::make_unique<std::vector<T>>(size, T(0));
    }
}

