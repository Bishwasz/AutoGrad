#include "tensor.h"     // Include the header file for the Tensor class declaration

#include <numeric>      // For std::accumulate, std::multiplies
#include <stdexcept>    // For std::runtime_error
#include <algorithm>    // For std::fill, std::all_of, std::any_of
#include <iostream>     // For potential std::cout debugging/info messages
#include <queue>        // For backward topological sort (if using queue)
#include <unordered_set>// For backward topological sort visited set
#include <vector>       // Needed again for implementation details
#include <random>


// --- Constructor Implementations --

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data, bool requires_grad)
    : shape_(shape), requires_grad_(requires_grad) {
    size_t size = compute_size();
    if (data.size() != size) {
        std::string shape_str;
        for (size_t i = 0; i < shape_.size(); ++i) {
            shape_str += std::to_string(shape_[i]) + (i == shape_.size() - 1 ? "" : ", ");
        }
        throw std::runtime_error("Data size (" + std::to_string(data.size()) + ") does not match shape [" + shape_str + "] which requires size " + std::to_string(size));
    }
    data_ = std::make_unique<std::vector<float>>(data);
    if (requires_grad_) {
        grad_ = std::make_unique<std::vector<float>>(size, 0.0f);
    }
}

Tensor::Tensor(const std::vector<int>& shape, bool requires_grad)
    : shape_(shape), requires_grad_(requires_grad) {
    for (int dim : shape_) {
        if (dim < 0) {
            throw std::runtime_error("Negative dimension in shape");
        }
    }
    size_t size = compute_size();
    data_ = std::make_unique<std::vector<float>>(size, 0.0f);
    std::random_device rd;  // random seed
    std::mt19937 gen(rd()); // Mersenne Twister RNG

    std::uniform_real_distribution<> dist(0.0, 1.0); 
    for (size_t i = 0; i < size; ++i) {
        (*data_)[i] = dist(gen); // Initialize data to zero
    }
    if (requires_grad_) {
        grad_ = std::make_unique<std::vector<float>>(size, 0.0f);
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
std::vector<float>& Tensor::data() {
    if (!data_) {
        throw std::runtime_error("Tensor data is null (non-const access)");
    }
    return *data_;
}
void Tensor::add_dependency(std::shared_ptr<Tensor> dep) {
    if (dep) {
        dependencies_.push_back(dep);
    }
}

void Tensor::set_backward_fn(std::function<void()> fn) {
    backward_fn_ = std::move(fn);
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
void Tensor::one_grad() {
    if (requires_grad_ && grad_) {
        std::fill(grad_->begin(), grad_->end(), 1.0f);
    }
}
void Tensor::set_data(const std::vector<float>& data){
    if (data.size() != compute_size()) {
        throw std::invalid_argument("Data size does not match tensor shape");
    } 
 
    *data_ = data;
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
    self_shared->one_grad();

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
    // for (const auto& node : topo_order) {
    //     if (node && node->backward_fn_) {
    //         node->backward_fn_();
    //     }
    // }
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        const auto& node = *it; // Get the shared_ptr from the reverse iterator
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
std::shared_ptr<Tensor> Tensor::broadcast_add(const Tensor& other) const {
    // Ensure both tensors are 2D
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::runtime_error("Broadcast addition requires 2D tensors");
    }

    // Check shapes: [M, N] + [1, N]
    size_t M = shape_[0];
    size_t N = shape_[1];
    if (other.shape_[0] != 1 || other.shape_[1] != N) {
        throw std::runtime_error("Incompatible shapes for broadcast addition: [" +
                                 std::to_string(M) + "," + std::to_string(N) + "] vs [" +
                                 std::to_string(other.shape_[0]) + "," + std::to_string(other.shape_[1]) + "]");
    }

    // Create result tensor with same shape as input [M, N]
    auto result = std::make_shared<Tensor>(shape_, requires_grad_ || other.requires_grad_);
    auto& result_data = *(result->data_);
    const auto& this_data = *data_;
    const auto& other_data = *other.data_;

    // Perform broadcast addition: add other[0, j] to each this[i, j]
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result_data[i * N + j] = this_data[i * N + j] + other_data[j];
        }
    }

    // Set up gradients if required
    if (result->requires_grad_) {
        auto this_shared = std::const_pointer_cast<Tensor>(shared_from_this());
        auto other_shared = std::const_pointer_cast<Tensor>(other.shared_from_this());
        result->dependencies_.push_back(this_shared);
        result->dependencies_.push_back(other_shared);

        result->backward_fn_ = [result_wptr = std::weak_ptr<Tensor>(result),
                               this_wptr = std::weak_ptr<Tensor>(this_shared),
                               other_wptr = std::weak_ptr<Tensor>(other_shared)]() {
            auto result_locked = result_wptr.lock();
            auto this_tensor = this_wptr.lock();
            auto other_tensor = other_wptr.lock();
            if (!result_locked || !this_tensor || !other_tensor) return;

            const auto& result_grad = result_locked->grad();
            size_t M = this_tensor->shape()[0];
            size_t N = this_tensor->shape()[1];

            // Gradient for this_tensor: pass through gradients
            if (this_tensor->requires_grad()) {
                auto& this_grad = this_tensor->grad();
                for (size_t i = 0; i < M; ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        if (i * N + j < this_grad.size()) {
                            this_grad[i * N + j] += result_grad[i * N + j];
                        }
                    }
                }
            }

            // Gradient for other_tensor: sum gradients across rows
            if (other_tensor->requires_grad()) {
                auto& other_grad = other_tensor->grad();
                for (size_t j = 0; j < N; ++j) {
                    float grad_sum = 0.0f;
                    for (size_t i = 0; i < M; ++i) {
                        grad_sum += result_grad[i * N + j];
                    }
                    if (j < other_grad.size()) {
                        other_grad[j] += grad_sum;
                    }
                }
            }
        };
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::log() const {
    // Compute log(data) element-wise
    std::vector<float> log_data(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        if (data_[i] <= 0.0f) {
            // Handle non-positive values for numerical stability
            log_data[i] = std::log(1e-15f); // Small constant to avoid log(0) or log(negative)
        } else {
            log_data[i] = std::log(data_[i]);
        }
    }

    // Create output tensor with same shape
    auto result = std::make_shared<Tensor>(shape_, log_data, requires_grad_);

    if (requires_grad_) {
        // Set up backward function for autograd
        // Gradient of log(x) is 1/x
        auto self = std::make_shared<Tensor>(*this); // Copy of this tensor
        result->backward_ = [self, result]() {
            // Ensure gradients are initialized
            if (self->grad_.empty()) {
                self->grad_.resize(self->data_.size(), 0.0f);
            }
            // Compute gradient: dL/dx = dL/dy * dy/dx, where dy/dx = 1/x
            const auto& result_grad = result->grad_;
            const auto& self_data = self->data_;
            for (size_t i = 0; i < self->grad_.size(); ++i) {
                float dx = (self_data[i] > 0.0f) ? (1.0f / self_data[i]) : 0.0f; // dy/dx = 1/x
                self->grad_[i] += result_grad[i] * dx; // Chain rule: dL/dx += dL/dy * dy/dx
            }
        };
        // Optional: Store parents for graph tracking (if your autograd system requires it)
        result->parents_ = {self};
    }

    return result;
}
std::shared_ptr<Tensor> Tensor::neg() const {
    std::vector<float> neg_data(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        neg_data[i] = -data_[i];
    }
    auto result = std::make_shared<Tensor>(shape_, neg_data, requires_grad_);
    if (requires_grad_) {
        auto self = std::make_shared<Tensor>(*this);
        result->backward_ = [self, result]() {
            if (self->grad_.empty()) {
                self->grad_.resize(self->data_.size(), 0.0f);
            }
            const auto& result_grad = result->grad_;
            for (size_t i = 0; i < self->grad_.size(); ++i) {
                self->grad_[i] += -result_grad[i]; // dL/dx = -dL/dy
            }
        };
        result->parents_ = {self};
    }
    return result;
}