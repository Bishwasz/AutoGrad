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

    size_t offset = 0;
    size_t stride = 1;
    for (int i = this->shape_.size() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= this->shape_[i])
            throw std::out_of_range("Index out of bounds");

        offset += indices[i] * stride;
        stride *= this->shape_[i];
    }

    return (*this->grad_)[offset];
}


template<typename T>
bool AutogradTensor<T>::requires_grad() const {
    return requires_grad_;
}
template<typename T>
void AutogradTensor<T>::set_requires_grad(bool req_grad) {
    requires_grad_ = req_grad;
    if (requires_grad_ && !grad_) {
         grad_ = std::make_unique<std::vector<T>>(this->BaseTensor<T>::compute_size(), T{0});
    } else if (!requires_grad_) {
        grad_.reset();
    }
}
template<typename T>
void AutogradTensor<T>::zero_grad() {
    if (requires_grad_) {
        if (!grad_) {
            grad_ = std::make_unique<std::vector<T>>(this->BaseTensor<T>::compute_size(), T{0});
        }
        std::fill(grad_->begin(), grad_->end(), T{0});
    }
}


template<typename T>
void AutogradTensor<T>::one_grad() {
    if (requires_grad_) {
        if (!grad_) {
            grad_ = std::make_unique<std::vector<T>>(this->BaseTensor<T>::compute_size(), T{1});
        }
        std::fill(grad_->begin(), grad_->end(), T{1});
    }
}

template<typename T>
void AutogradTensor<T>::add_dependency(AutogradTensor<T>* dep) {
    dependencies_.push_back(dep);
}


template<typename T>
void AutogradTensor<T>::set_backward_fn(std::function<void()> fn) {
    backward_fn_ = std::move(fn);
}