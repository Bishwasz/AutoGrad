#include "tensor.h"     // Include the header file for the Tensor class declaration

#include <numeric>      // For std::accumulate, std::multiplies
#include <stdexcept>    // For std::runtime_error
#include <algorithm>    // For std::fill, std::all_of, std::any_of
#include <iostream>     // For potential std::cout debugging/info messages
#include <queue>        // For backward topological sort (if using queue)
#include <unordered_set>// For backward topological sort visited set
#include <vector>       // Needed again for implementation details

// --- Constructor Implementations ---
Tensor::Tensor(const std::vector<int>& shape, bool requires_grad)
    : shape_(shape), requires_grad_(requires_grad) {
    size_t size = compute_size();
    data_ = std::make_unique<std::vector<float>>(size, 0.0f);
    if (requires_grad_) {
        grad_ = std::make_unique<std::vector<float>>(size, 0.0f);
    }
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data, bool requires_grad)
    : shape_(shape), requires_grad_(requires_grad) {
    data_ = std::make_unique<std::vector<float>>(data);
    if (data_->size() != compute_size()) {
        std::string shape_str;
        for(size_t i=0; i<shape_.size(); ++i) shape_str += std::to_string(shape_[i]) + (i == shape_.size()-1 ? "" : ", ");
        throw std::runtime_error("Data size (" + std::to_string(data_->size()) + ") does not match shape [" + shape_str + "] which requires size " + std::to_string(compute_size()));
    }
    if (requires_grad_) {
        grad_ = std::make_unique<std::vector<float>>(data_->size(), 0.0f);
    }
}

// --- Getter Implementations ---
size_t Tensor::compute_size() const {
     if (shape_.empty()) return 1; // Scalar/empty shape case
    // Ensure size_t for accumulation start value
    return std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>());
}

const std::vector<float>& Tensor::data() const {
     if (!data_) { throw std::runtime_error("Tensor data is null"); }
    return *data_;
}

// Const grad getter
const std::vector<float>& Tensor::grad() const {
    if (!requires_grad_) {
        throw std::runtime_error("Tensor does not require gradients (const access)");
    }
     if (!grad_) {
        throw std::runtime_error("Gradient requested but not allocated (const access)");
    }
    return *grad_;
}

// Non-const grad getter (with lazy allocation)
std::vector<float>& Tensor::grad() {
    if (!requires_grad_) {
        throw std::runtime_error("Tensor does not require gradients (non-const access)");
    }
    if (!grad_) {
        grad_ = std::make_unique<std::vector<float>>(compute_size(), 0.0f);
    }
    return *grad_;
}

const std::vector<int>& Tensor::shape() const {
    return shape_;
}

bool Tensor::requires_grad() const {
    return requires_grad_;
}

// --- Setter / Modifier Implementations ---
void Tensor::set_requires_grad(bool req_grad) {
    requires_grad_ = req_grad;
    if (requires_grad_ && !grad_) {
         grad_ = std::make_unique<std::vector<float>>(compute_size(), 0.0f);
    } else if (!requires_grad_) {
        grad_.reset();
    }
}

void Tensor::zero_grad() {
    if (requires_grad_ && grad_) {
        std::fill(grad_->begin(), grad_->end(), 0.0f);
    }
}

// --- Autograd Implementation ---
void Tensor::backward() {
    std::shared_ptr<Tensor> self_shared;
    try {
         self_shared = shared_from_this();
    } catch (const std::bad_weak_ptr& e) {
        throw std::runtime_error("backward() must be called on a Tensor managed by std::shared_ptr");
    }

    if (!requires_grad_) { return; }

    // Initialize gradient if needed
    if (compute_size() == 1) {
        auto& g = grad();
        if (g.empty()) { g.resize(1, 0.0f); }
        if (g[0] == 0.0f) { g[0] = 1.0f; }
    } else if (!grad_) {
         // Ensure gradient exists for non-scalars if requires_grad is true
         grad(); // Call non-const version to allocate
    } else {
        // Optional warning for non-scalar zero gradients
        // bool all_zero = std::all_of(grad_->begin(), grad_->end(), [](float g){ return g == 0.0f; });
        // if(all_zero) { /* warning */ }
    }

    // Build topological order
    std::unordered_set<std::shared_ptr<Tensor>> visited;
    std::vector<std::shared_ptr<Tensor>> topo_order;
    std::function<void(std::shared_ptr<Tensor>)> build_topo =
        [&](std::shared_ptr<Tensor> node) {
        if (!node || visited.count(node)) { return; }
        visited.insert(node);
        // Explore dependencies regardless of their requires_grad status,
        // as they form the graph structure. The check happens in accumulation.
        for (auto& dep : node->dependencies_) {
            build_topo(dep);
        }
        topo_order.push_back(node);
    };

    build_topo(self_shared);

    // Execute backward functions
    for (const auto& node : topo_order) {
        if (node && node->backward_fn_) {
            node->backward_fn_();
        }
    }
}

// --- Operator Implementations ---
std::shared_ptr<Tensor> Tensor::operator+(const Tensor& other) const {
     if (shape_ != other.shape_) {
         std::string shape1_str, shape2_str;
         for(size_t i=0; i<shape_.size(); ++i) shape1_str += std::to_string(shape_[i]) + (i == shape_.size()-1 ? "" : ", ");
         for(size_t i=0; i<other.shape_.size(); ++i) shape2_str += std::to_string(other.shape_[i]) + (i == other.shape_.size()-1 ? "" : ", ");
        throw std::runtime_error("Shape mismatch for addition: [" + shape1_str + "] vs [" + shape2_str + "]");
    }

    auto result = std::make_shared<Tensor>(shape_, requires_grad_ || other.requires_grad_);
    size_t size = compute_size();
    for (size_t i = 0; i < size; ++i) {
        (*(result->data_))[i] = (*data_)[i] + (*other.data_)[i];
    }

    if (result->requires_grad_) {
        auto this_shared = std::const_pointer_cast<Tensor>(shared_from_this());
        auto other_shared = std::const_pointer_cast<Tensor>(other.shared_from_this());
        result->dependencies_.push_back(this_shared);
        result->dependencies_.push_back(other_shared);

        result->backward_fn_ = [result_wptr = std::weak_ptr<Tensor>(result),
                                this_wptr = std::weak_ptr<Tensor>(this_shared),
                                other_wptr = std::weak_ptr<Tensor>(other_shared)
                               ]() {
            auto result_locked = result_wptr.lock();
            auto this_tensor = this_wptr.lock();
            auto other_tensor = other_wptr.lock();
            if (!result_locked || !this_tensor || !other_tensor) return;

            if (this_tensor->requires_grad()) {
                 auto& this_grad = this_tensor->grad();
                 const auto& result_grad = result_locked->grad();
                for (size_t i = 0; i < this_tensor->compute_size(); ++i) {
                     if (i < result_grad.size()) { this_grad[i] += result_grad[i]; }
                }
            }
            if (other_tensor->requires_grad()) {
                auto& other_grad = other_tensor->grad();
                const auto& result_grad = result_locked->grad();
                for (size_t i = 0; i < other_tensor->compute_size(); ++i) {
                     if (i < result_grad.size()) { other_grad[i] += result_grad[i]; }
                }
            }
        };
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::operator*(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::runtime_error("Matrix multiplication requires 2D tensors");
    }
    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t N = other.shape_[1];
     if (K != static_cast<size_t>(other.shape_[0])) {
        throw std::runtime_error("Incompatible dimensions for matrix multiplication: ["
             + std::to_string(M) + "," + std::to_string(K) + "] * ["
             + std::to_string(other.shape_[0]) + "," + std::to_string(N) + "]");
    }

    std::vector<int> result_shape = {static_cast<int>(M), static_cast<int>(N)};
    auto result = std::make_shared<Tensor>(result_shape, requires_grad_ || other.requires_grad_);

    const auto& this_data = *data_;
    const auto& other_data = *other.data_;
    auto& result_data = *(result->data_);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += this_data[i * K + k] * other_data[k * N + j];
            }
            result_data[i * N + j] = sum;
        }
    }

    if (result->requires_grad_) {
        auto this_shared = std::const_pointer_cast<Tensor>(shared_from_this());
        auto other_shared = std::const_pointer_cast<Tensor>(other.shared_from_this());
        result->dependencies_.push_back(this_shared);
        result->dependencies_.push_back(other_shared);

        result->backward_fn_ = [result_wptr = std::weak_ptr<Tensor>(result),
                                this_wptr = std::weak_ptr<Tensor>(this_shared),
                                other_wptr = std::weak_ptr<Tensor>(other_shared)
                               ]() {
            auto result_locked = result_wptr.lock();
            auto this_tensor = this_wptr.lock();
            auto other_tensor = other_wptr.lock();
            if (!result_locked || !this_tensor || !other_tensor) return;

            const auto& this_data = this_tensor->data();
            const auto& other_data = other_tensor->data();
            const auto& result_grad = result_locked->grad();

            size_t M = this_tensor->shape()[0];
            size_t K = this_tensor->shape()[1];
            size_t N = other_tensor->shape()[1];

            if (this_tensor->requires_grad()) {
                 auto& this_grad = this_tensor->grad();
                for (size_t i = 0; i < M; ++i) {
                    for (size_t k = 0; k < K; ++k) {
                        float grad_sum = 0.0f;
                        for (size_t j = 0; j < N; ++j) {
                            grad_sum += result_grad[i * N + j] * other_data[k * N + j];
                        }
                         if ((i * K + k) < this_grad.size()) { this_grad[i * K + k] += grad_sum; }
                    }
                }
            }
            if (other_tensor->requires_grad()) {
                 auto& other_grad = other_tensor->grad();
                for (size_t k = 0; k < K; ++k) {
                    for (size_t j = 0; j < N; ++j) {
                        float grad_sum = 0.0f;
                        for (size_t i = 0; i < M; ++i) {
                            grad_sum += this_data[i * K + k] * result_grad[i * N + j];
                        }
                         if ((k * N + j) < other_grad.size()) { other_grad[k * N + j] += grad_sum; }
                    }
                }
            }
        };
    }
    return result;
}