#include "AutoGradTensor.h"

template <typename T>
T& AutogradTensor<T>::at(const std::vector<int>& indices) {
    if (indices.size() != this->shape_.size())
        throw std::invalid_argument("Incorrect number of indices");

    size_t offset = 0;
    size_t stride = 1;
    for (int i = this->shape_.size() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= this->shape_[i])
            throw std::out_of_range("Index out of bounds");

        offset += indices[i] * stride;
        stride *= this->shape_[i];
    }

    return (*this->data_)[offset];
}

template <typename T>
const T& AutogradTensor<T>::at(const std::vector<int>& indices) const {
    if (indices.size() != this->shape_.size())
        throw std::invalid_argument("Incorrect number of indices");

    size_t offset = 0;
    size_t stride = 1;
    for (int i = this->shape_.size() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= this->shape_[i])
            throw std::out_of_range("Index out of bounds");

        offset += indices[i] * stride;
        stride *= this->shape_[i];
    }

    return (*this->data_)[offset];
}

template <typename T>
const T& AutogradTensor<T>::at_grad(const std::vector<int>& indices) const {
    if (indices.size() != this->shape_.size())
        throw std::invalid_argument("Incorrect number of indices");

    return grad_.at(indices);
}

template<typename T>
bool AutogradTensor<T>::requires_grad() const {
    return requires_grad_;
}

// CORRECTED: This function now correctly handles the grad_ member object.
template<typename T>
void AutogradTensor<T>::set_requires_grad(bool req_grad) {
    requires_grad_ = req_grad;
    if (requires_grad_) {
        // If the grad tensor's shape doesn't match, re-initialize it.
        if (grad_.shape() != this->shape_) {
            grad_ = BaseTensor<T>(this->shape_); // Re-assign with correct shape
        }
        grad_.zero_data(); // Always zero out gradients when enabling
    }
}

template<typename T>
void AutogradTensor<T>::zero_grad() {
    if (requires_grad_) {
        grad_.zero_data();
    }
}

template<typename T>
void AutogradTensor<T>::one_grad() {
    if (requires_grad_) {
        grad_.one_data();
    }
}

template<typename T>
void AutogradTensor<T>::add_dependency(AutogradTensor<T>* dep) {
    if (dep != nullptr) {
        dependencies_.push_back(dep);
    } else {
        std::cerr << "Warning: Attempted to add null dependency" << std::endl;
    }
}

// CORRECTED: The signature now matches the header and the logic for the memory fix.
template<typename T>
void AutogradTensor<T>::set_backward_fn(BackwardFn<T> fn) {
    backward_fn_ = std::move(fn);
}