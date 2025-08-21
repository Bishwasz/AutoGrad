#include <random>
#include <iostream>
#include <functional>
#include <memory>
#include <unordered_set>
#include <algorithm>
#include <cstdio>
#include "../BaseTensor/Mul.h" // Assuming this contains openBlasMultiply

// Forward declaration if AutogradTensor is in its own header
template<typename T>
class AutogradTensor;

// CORRECTED: The backward function now accepts a pointer to its owner tensor.
// This is the core fix for the memory safety issue.
template<typename T>
using BackwardFn = std::function<void(AutogradTensor<T>*)>;


// Helper function for transposing a matrix
template<typename T>
void transpose(const std::vector<T>& input, std::vector<T>& output, size_t rows, size_t cols) {
    output.resize(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}


template<typename T>
void AutogradTensor<T>::backward() {
    if (!requires_grad_) return;

    this->grad_.one_data(); // Initialize gradient of the final tensor to ones

    std::unordered_set<AutogradTensor<T>*> visited;
    std::vector<AutogradTensor<T>*> topo_order;

    std::function<void(AutogradTensor<T>*)> build_topo = 
        [&](AutogradTensor<T>* node) {
        if (!node || visited.count(node)) return;
        visited.insert(node);

        for (auto* dep : node->dependencies_) {
            build_topo(dep);
        }
        topo_order.push_back(node);
    };

    build_topo(this);

    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        auto* node = *it;
        if (node && node->backward_fn_) {
            // CORRECTED: Pass the node itself to its backward function.
            node->backward_fn_(node);
        }
    }
}

// CORRECTED: Takes `other` by const reference.
template <typename T>
AutogradTensor<T> AutogradTensor<T>::operator+(const AutogradTensor<T>& other) {
    if (this->shape_ != other.shape()) {
        throw std::runtime_error("Shape mismatch for addition");
    }

    size_t size = this->compute_size();
    std::vector<T> result_data(size);
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = (*this->data_)[i] + other.data()[i];
    }

    bool result_requires_grad = requires_grad_ || other.requires_grad_;
    AutogradTensor<T> result(this->shape_, result_data, result_requires_grad);

    if (result_requires_grad) {
        result.add_dependency(this);
        // Note: The lifetime of `other` must exceed the call to backward().
        result.add_dependency(const_cast<AutogradTensor<T>*>(&other));

        // CORRECTED: Lambda accepts a `self` pointer, avoiding dangling references.
        result.set_backward_fn([this, &other](AutogradTensor<T>* self) {
            const auto& grad_data = self->grad().data();

            if (this->requires_grad()) {
                auto& this_grad = this->grad().data();
                for (size_t i = 0; i < grad_data.size(); ++i) {
                    this_grad[i] += grad_data[i];
                }
            }

            if (other.requires_grad()) {
                auto& other_grad = const_cast<AutogradTensor<T>&>(other).grad().data();
                for (size_t i = 0; i < grad_data.size(); ++i) {
                    other_grad[i] += grad_data[i];
                }
            }
        });
    }
    return result;
}

// CORRECTED: Takes `other` by const reference.
template <typename T>
AutogradTensor<T> AutogradTensor<T>::operator-(const AutogradTensor<T>& other) {
    if (this->shape_ != other.shape()) {
        throw std::runtime_error("Shape mismatch for subtraction");
    }

    std::vector<T> result_data(this->compute_size());
    for (size_t i = 0; i < this->compute_size(); ++i) {
        result_data[i] = (*this->data_)[i] - other.data()[i];
    }

    bool result_requires_grad = requires_grad_ || other.requires_grad_;
    AutogradTensor<T> result(this->shape_, result_data, result_requires_grad);

    if (result_requires_grad) {
        result.add_dependency(this);
        result.add_dependency(const_cast<AutogradTensor<T>*>(&other));

        // CORRECTED: Lambda accepts a `self` pointer.
        result.set_backward_fn([this, &other](AutogradTensor<T>* self) {
            const auto& result_grad = self->grad().data();

            if (this->requires_grad()) {
                auto& this_grad = this->grad().data();
                for (size_t i = 0; i < result_grad.size(); ++i) {
                    this_grad[i] += result_grad[i];
                }
            }

            if (other.requires_grad()) {
                auto& other_grad = const_cast<AutogradTensor<T>&>(other).grad().data();
                for (size_t i = 0; i < result_grad.size(); ++i) {
                    other_grad[i] -= result_grad[i];
                }
            }
        });
    }
    return result;
}

// CORRECTED: Takes `other` by const reference.
template <typename T>
AutogradTensor<T> AutogradTensor<T>::operator*(const AutogradTensor<T>& other) {
    if (this->shape_[1] != other.shape()[0]) {
        throw std::runtime_error("Shape mismatch for multiplication.");
    }

    size_t M = this->shape_[0];
    size_t K = this->shape_[1];
    size_t N = other.shape()[1];
    std::vector<T> result_data(M * N);

    openBlasMultiply(this->data(), other.data(), result_data, M, K, N);

    bool result_requires_grad = this->requires_grad_ || other.requires_grad_;
    AutogradTensor<T> result({static_cast<int>(M), static_cast<int>(N)}, result_data, result_requires_grad);

    if (result_requires_grad) {
        result.add_dependency(this);
        result.add_dependency(const_cast<AutogradTensor<T>*>(&other));

        // IMPROVED: Capture data by value for safety, as only original values are needed.
        auto this_data_copy = *this->data_;
        auto other_data_copy = *other.data_;

        result.set_backward_fn([this, &other, this_data_copy, other_data_copy, M, K, N]
                               (AutogradTensor<T>* self) {
            const auto& result_grad = self->grad().data();

            if (this->requires_grad()) {
                // Grad for `this` is result_grad * other^T
                std::vector<T> other_T;
                transpose(other_data_copy, other_T, K, N);
                
                std::vector<T> this_grad_update(M * K);
                openBlasMultiply(result_grad, other_T, this_grad_update, M, N, K);
                
                auto& this_grad = this->grad().data();
                for(size_t i = 0; i < this_grad.size(); ++i) {
                    this_grad[i] += this_grad_update[i];
                }
            }

            if (other.requires_grad()) {
                // Grad for `other` is this^T * result_grad
                std::vector<T> this_T;
                transpose(this_data_copy, this_T, M, K);

                std::vector<T> other_grad_update(K * N);
                openBlasMultiply(this_T, result_grad, other_grad_update, K, M, N);

                auto& other_grad = const_cast<AutogradTensor<T>&>(other).grad().data();
                for(size_t i = 0; i < other_grad.size(); ++i) {
                    other_grad[i] += other_grad_update[i];
                }
            }
        });
    }
    return result;
}

template <typename T>
AutogradTensor<T> AutogradTensor<T>::log() {
    std::vector<T> result_data(this->compute_size());
    for (size_t i = 0; i < result_data.size(); ++i) {
        if ((*this->data_)[i] <= T{0}) throw std::runtime_error("Log of non-positive value");
        result_data[i] = std::log((*this->data_)[i]);
    }
    AutogradTensor<T> result(this->shape_, result_data, requires_grad_);
    if (requires_grad_) {
        result.add_dependency(this);
        // CORRECTED: Lambda accepts a `self` pointer.
        result.set_backward_fn([this](AutogradTensor<T>* self) {
            const auto& result_grad = self->grad();
            auto& this_grad = this->grad();
            for (size_t i = 0; i < this->compute_size(); ++i) {
                this_grad[i] += result_grad[i] / (*this->data_)[i];
            }
        });
    }
    return result;
}

// CORRECTED: Takes `other` by const reference.
template <typename T>
AutogradTensor<T> AutogradTensor<T>::broadcast_add(const AutogradTensor<T>& other) {
    if (this->shape_.size() != 2 || other.shape().size() != 1 ||
        this->shape_[1] != other.shape()[0]) {
        throw std::runtime_error("Shape mismatch for broadcast addition");
    }

    const size_t M = this->shape_[0];
    const size_t N = this->shape_[1];
    std::vector<T> result_data(M * N);
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            result_data[r * N + c] = (*this->data_)[r * N + c] + other.data()[c];
        }
    }

    bool result_requires_grad = this->requires_grad_ || other.requires_grad_;
    AutogradTensor<T> result(this->shape_, result_data, result_requires_grad);

    if (result_requires_grad) {
        result.add_dependency(this);
        result.add_dependency(const_cast<AutogradTensor<T>*>(&other));

        // CORRECTED: Lambda accepts a `self` pointer.
        result.set_backward_fn([this, &other, M, N](AutogradTensor<T>* self) {
            const auto& grad_out = self->grad().data();

            if (this->requires_grad()) {
                auto& this_grad = this->grad().data();
                for (size_t i = 0; i < M * N; ++i) {
                    this_grad[i] += grad_out[i];
                }
            }
            if (other.requires_grad()) {
                auto& other_grad = const_cast<AutogradTensor<T>&>(other).grad().data();
                for (size_t c = 0; c < N; ++c) {
                    T grad_sum = T{0};
                    for (size_t r = 0; r < M; ++r) {
                        grad_sum += grad_out[r * N + c];
                    }
                    other_grad[c] += grad_sum;
                }
            }
        });
    }
    return result;
}

template <typename T>
AutogradTensor<T> AutogradTensor<T>::relu() {
    size_t size = this->compute_size();
    std::vector<T> result_data(size);
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = std::max((*this->data_)[i], T{0});
    }

    AutogradTensor<T> result(this->shape_, result_data, requires_grad_);

    if (requires_grad_) {
        result.add_dependency(this);

        // CORRECTED: Lambda accepts a `self` pointer.
        result.set_backward_fn([this, size](AutogradTensor<T>* self) {
            const auto& grad_out = self->grad().data();
            auto& grad_in = this->grad().data();

            for (size_t i = 0; i < size; ++i) {
                if ((*this->data_)[i] > T{0}) {
                    grad_in[i] += grad_out[i];
                }
            }
        });
    }
    return result;
}