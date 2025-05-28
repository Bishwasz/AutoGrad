
#include <random>
#include <functional>
#include <memory>
#include <unordered_set>
#include <algorithm> // for std::max_element



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
        std::uniform_int_distribution<T> dist(0, 100);  // Adjust range as needed
        this->grad_ = std::make_unique<std::vector<int>>(this->BaseTensor<T>::compute_size(), 0);
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


template<typename T>
void AutogradTensor<T>::backward() {
    if (!requires_grad_) return;

    this->one_grad();

    std::unordered_set<AutogradTensor<T>*, std::hash<AutogradTensor<T>*>, std::equal_to<AutogradTensor<T>*>> visited;
    std::vector<AutogradTensor<T>*> topo_order;

    // Fix: Use pointer type for node
    std::function<void(AutogradTensor<T>*)> build_topo = [&](AutogradTensor<T>* node) {
        if (!node || visited.count(node)) return;
        visited.insert(node);

        for (auto* dep : node->dependencies_) {
            build_topo(dep);
        }
        topo_order.push_back(node);
    };

    build_topo(this);

    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        const auto* node = *it; // Use pointer type
        if (node && node->backward_fn_) {
            node->backward_fn_();
        }
    }
}
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
AutogradTensor<T> AutogradTensor<T>::operator+(AutogradTensor<T>& other) {
    if (this->shape_ != other.shape()) {
        throw std::runtime_error("Shape mismatch for addition");
    }

    std::vector<T> result_data(this->compute_size());
    for (size_t i = 0; i <this->compute_size(); ++i) {
        result_data[i] = (*this->data_)[i] + other.data()[i];
    }

    bool result_requires_grad = requires_grad_ || other.requires_grad_;
    AutogradTensor<T> result(this->shape_, result_data, result_requires_grad);

    if (result_requires_grad) {
        result.add_dependency(this);
        result.add_dependency(&other);

        result.set_backward_fn([this, &other, &result]() {
            const auto& result_grad = result.grad();

            if (this->requires_grad()) {
                auto& this_grad = this->grad();
                for (size_t i = 0; i < result.compute_size(); ++i) {
                    this_grad[i] += result_grad[i];
                }
            }

            if (other.requires_grad()) {
                auto& other_grad = other.grad();
                for (size_t i = 0; i < result.compute_size(); ++i) {
                    other_grad[i] += result_grad[i];
                }
            }
        });
    }

    return result;
}
template <typename T>
AutogradTensor<T> AutogradTensor<T>::operator-(AutogradTensor<T>& other) {
    if (this->shape_ != other.shape()) {
        throw std::runtime_error("Shape mismatch for addition");
    }

    std::vector<T> result_data(this->compute_size());
    for (size_t i = 0; i <this->compute_size(); ++i) {
        result_data[i] = (*this->data_)[i] - other.data()[i];
    }

    bool result_requires_grad = requires_grad_ || other.requires_grad_;
    AutogradTensor<T> result(this->shape_, result_data, result_requires_grad);

    if (result_requires_grad) {
        result.add_dependency(this);
        result.add_dependency(&other);

        result.set_backward_fn([this, &other, &result]() {
            const auto& result_grad = result.grad();

            if (this->requires_grad()) {
                auto& this_grad = this->grad();
                for (size_t i = 0; i < result.compute_size(); ++i) {
                    this_grad[i] += result_grad[i];
                }
            }

            if (other.requires_grad()) {
                auto& other_grad = other.grad();
                for (size_t i = 0; i < result.compute_size(); ++i) {
                    other_grad[i] -= result_grad[i];
                }
            }
        });
    }

    return result;
}



template <typename T>
AutogradTensor<T> AutogradTensor<T>::operator*(AutogradTensor<T>& other)  {
    if (this->shape_[1] != other.shape()[0]) {
        throw std::runtime_error("Shape mismatch for multiplication: incompatible inner dimensions.");
    }
    int M = this->shape_[0];
    int K = this->shape_[1];
    int N = this->shape_[1];
    std::vector<T> result_data(M * N, T(0));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += (*this->data_)[i * K + k] * other.data()[k * N + j];
            }
            result_data[i * N + j] = sum;
        }
    }
    AutogradTensor<T> result(this->shape_, result_data, requires_grad_ || other.requires_grad_);
    if (result.requires_grad_) {
        result.add_dependency(this);
        result.add_dependency(&other);

        result.set_backward_fn([this, &other, &result]() {
            const auto& result_grad = result.grad();

            if (this->requires_grad()) {
                auto& this_grad = this->grad();
                for (int i = 0; i < this->shape_[0]; ++i) {
                    for (int k = 0; k < this->shape_[1]; ++k) {
                        T grad_val = 0;
                        for (int j = 0; j < other.shape()[1]; ++j) {
                            grad_val += result_grad[i * other.shape()[1] + j] * other.data()[k * other.shape()[1] + j];
                        }
                        this_grad[i * this->shape_[1] + k] += grad_val;
                    }
                }
            }

            if (other.requires_grad()) {
                auto& other_grad = other.grad();
                for (int k = 0; k < other.shape()[0]; ++k) {
                    for (int j = 0; j < other.shape()[1]; ++j) {
                        T grad_val = 0;
                        for (int i = 0; i < this->shape_[0]; ++i) {
                            grad_val += (*this->data_)[i * this->shape_[1] + k] * result_grad[i * other.shape()[1] + j];
                        }
                        other_grad[k * other.shape()[1] + j] += grad_val;
                    }
                }
            }
        });
    }

    return result;
}





template <typename T>
AutogradTensor<T> AutogradTensor<T>::log()  {
    std::vector<T> result_data(this->BaseTensor<T>::compute_size());
    for (size_t i = 0; i < result_data.size(); ++i) {
        if ((*this->data_)[i] <= T{0}) throw std::runtime_error("Log of non-positive value");
        result_data[i] = std::log((*this->data_)[i]);
    }
    AutogradTensor<T> result(this->shape_, result_data, requires_grad_);
    if (requires_grad_) {
        result.add_dependency(this);
        result.set_backward_fn([this,&result]() {
            const auto& result_grad = result.grad();
            auto& this_grad = this->grad(); // Non-const grad()
            for (size_t i = 0; i < this->BaseTensor<T>::compute_size(); ++i) {
                this_grad[i] += result_grad[i] / (*this->data_)[i];
            }
        });
    }
    return result;


}

template <typename T>
AutogradTensor<T> AutogradTensor<T>::broadcast_add(AutogradTensor<T>& other) {
    if (this->shape_[1] != other.shape()[0]) {
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

    AutogradTensor<T> result(this->shape_, result_data, requires_grad_ || other.requires_grad_);
    if (result.requires_grad_) {
        result.add_dependency(this);
        result.add_dependency(&other);

        result.set_backward_fn([this, &other, &result]() {
            const auto& result_grad = result.grad();

            if (this->requires_grad_) {
                auto& this_grad = this->grad();
                for (int r = 0; r < this->shape_[0]; ++r) {
                    for (int c = 0; c < this->shape_[1]; ++c) {
                        this_grad[r * this->shape_[1] + c] += result_grad[r * this->shape_[1] + c];
                    }
                }
            }

            if (other.requires_grad_) {
                auto& other_grad = other.grad();
                for (int c = 0; c < other.shape()[0]; ++c) {
                    T grad_sum = 0;
                    for (int r = 0; r < this->shape_[0]; ++r) {
                        grad_sum += result_grad[r * this->shape_[1] + c];
                    }
                    other_grad[c] += grad_sum;
                }
            }
        });
    }


    return result;

}


template <typename T>
AutogradTensor<T> AutogradTensor<T>::cross_entropy_loss(AutogradTensor<T>& labels) {
    // Check shape compatibility
    if (this->shape_.size() != 2 || labels.shape_.size() != 2 || 
        this->shape_[0] != labels.shape_[0] || this->shape_[1] != labels.shape_[1]) {
        throw std::runtime_error("Shape mismatch for cross-entropy loss");
    }

    const size_t batch_size = this->shape_[0]; // N
    const size_t num_classes = this->shape_[1]; // C

    // Compute softmax and cross-entropy loss
    std::vector<T> softmax_probs(batch_size * num_classes);
    T loss_sum = 0.0;

    for (size_t n = 0; n < batch_size; ++n) {
        // Compute softmax for sample n
        T max_logit = (*this->data_)[n * num_classes]; // For numerical stability
        for (size_t i = 1; i < num_classes; ++i) {
            max_logit = std::max(max_logit, (*this->data_)[n * num_classes + i]);
        }

        T exp_sum = 0.0;
        for (size_t i = 0; i < num_classes; ++i) {
            softmax_probs[n * num_classes + i] = 
                std::exp((*this->data_)[n * num_classes + i] - max_logit);
            exp_sum += softmax_probs[n * num_classes + i];
        }

        for (size_t i = 0; i < num_classes; ++i) {
            softmax_probs[n * num_classes + i] /= exp_sum;
            // Compute loss contribution: -y_i * log(softmax(z_i))
            T prob = softmax_probs[n * num_classes + i];
            T label = labels.data()[n * num_classes + i];
            if (prob > 0 && label > 0) { // Avoid log(0) and ignore zero labels
                loss_sum -= label * std::log(prob);
            }
        }
    }

    // Average loss over batch
    T avg_loss = loss_sum / static_cast<T>(batch_size);
    std::vector<T> result_data = {avg_loss};
    AutogradTensor<T> result({1}, result_data, this->requires_grad_ || labels.requires_grad_);

    // Set up backward pass if gradients are required
    if (result.requires_grad_) {
        result.add_dependency(this);
        result.add_dependency(&labels);

        result.set_backward_fn([this, &labels, &result, softmax_probs, batch_size, num_classes]() {
            const auto& result_grad = result.grad(); // Scalar gradient (dL/dloss)
            T grad_scale = result_grad[0] / static_cast<T>(batch_size); // Scale by 1/N

            // Gradient w.r.t. logits: (softmax - labels) / batch_size
            if (this->requires_grad_) {
                auto& this_grad = this->grad();
                for (size_t n = 0; n < batch_size; ++n) {
                    for (size_t i = 0; i < num_classes; ++i) {
                        size_t idx = n * num_classes + i;
                        this_grad[idx] += grad_scale * 
                            (softmax_probs[idx] - labels.data()[idx]);
                    }
                }
            }

            // Gradient w.r.t. labels (rarely needed, but included for completeness)
            if (labels.requires_grad_) {
                auto& labels_grad = labels.grad();
                for (size_t n = 0; n < batch_size; ++n) {
                    for (size_t i = 0; i < num_classes; ++i) {
                        size_t idx = n * num_classes + i;
                        T prob = softmax_probs[idx];
                        if (prob > 0) {
                            labels_grad[idx] += grad_scale * (-std::log(prob));
                        }
                    }
                }
            }
        });
    }

    return result;
}

template <typename T>
AutogradTensor<T> AutogradTensor<T>::relu() {
    std::vector<T> result_data(this->BaseTensor<T>::compute_size());
    for (size_t i = 0; i < result_data.size(); ++i) {
        result_data[i] = std::max((*this->data_)[i], T{0});
    }
    AutogradTensor<T> result(this->shape_, result_data, requires_grad_);
    if (requires_grad_) {
        result.add_dependency(this);
        result.set_backward_fn([this,&result]() {
            const auto& result_grad = result.grad();
            auto& this_grad = this->grad(); // Non-const grad()
            for (size_t i = 0; i < this->BaseTensor<T>::compute_size(); ++i) {
                this_grad[i] += (*this->data_)[i] > T{0} ? result_grad[i] : T{0};
            }
        });
    }
    return result;
}

// template <typename T>
// AutogradTensor<T> AutogradTensor<T>::log() const {
//     std::vector<T> result_data(this->BaseTensor<T>::compute_size());
//     for (size_t i = 0; i < result_data.size(); ++i) {
//         if ((*this->data_)[i] <= T{0}) throw std::runtime_error("Log of non-positive value");
//         result_data[i] = std::log((*this->data_)[i]);
//     }
//     AutogradTensor<T> result(this->shape_, result_data, requires_grad_);
//     if (requires_grad_) {
//         result.add_dependency(const_cast<AutogradTensor<T>*>(this));
//         result.set_backward_fn([this_ptr = const_cast<AutogradTensor<T>*>(this),
//                                result_ptr = std::make_shared<AutogradTensor<T>>(result)]() {
//             const auto& result_grad = result_ptr->grad();
//             auto& this_grad = this_ptr->grad(); // Non-const grad()
//             for (size_t i = 0; i < this_ptr->BaseTensor<T>::compute_size(); ++i) {
//                 this_grad[i] += result_grad[i] / (*this_ptr->data_)[i];
//             }
//         });
//     }
//     return result;
// }

// template <typename T>
// AutogradTensor<T> AutogradTensor<T>::neg() const {
//     std::vector<T> result_data(this->BaseTensor<T>::compute_size());
//     for (size_t i = 0; i < result_data.size(); ++i) {
//         result_data[i] = -(*this->data_)[i];
//     }
//     AutogradTensor<T> result(this->shape_, result_data, requires_grad_);
//     if (requires_grad_) {
//         result.add_dependency(const_cast<AutogradTensor<T>*>(this));
//         result.set_backward_fn([this_ptr = const_cast<AutogradTensor<T>*>(this),
//                                result_ptr = std::make_shared<AutogradTensor<T>>(result)]() {
//             const auto& result_grad = result_ptr->grad();
//             auto& this_grad = this_ptr->grad();
//             for (size_t i = 0; i < this_ptr->BaseTensor<T>::compute_size(); ++i) {
//                 this_grad[i] += -result_grad[i];
//             }
//         });
//     }
//     return result;
// }



// template <typename T>
// AutogradTensor<T> AutogradTensor<T>::sigmoid() const {
//     std::vector<T> result_data(this->BaseTensor<T>::compute_size());
//     for (size_t i = 0; i < result_data.size(); ++i) {
//         result_data[i] = T{1} / (T{1} + std::exp(-(*this->data_)[i]));
//     }
//     AutogradTensor<T> result(this->shape_, result_data, requires_grad_);
//     if (requires_grad_) {
//         result.add_dependency(const_cast<AutogradTensor<T>*>(this));
//         result.set_backward_fn([this_ptr = const_cast<AutogradTensor<T>*>(this),
//                                result_ptr = std::make_shared<AutogradTensor<T>>(result)]() {
//             const auto& result_grad = result_ptr->grad();
//             auto& this_grad = this_ptr->grad();
//             for (size_t i = 0; i < this_ptr->BaseTensor<T>::compute_size(); ++i) {
//                 T sigmoid_val = T{1} / (T{1} + std::exp(-(*this_ptr->data_)[i]));
//                 this_grad[i] += result_grad[i] * sigmoid_val * (T{1} - sigmoid_val);
//             }
//         });
//     }
//     return result;
// }

// template <typename T>
// AutogradTensor<T> AutogradTensor<T>::relu() const {
//     std::vector<T> result_data(this->BaseTensor<T>::compute_size());
//     for (size_t i = 0; i < result_data.size(); ++i) {
//         result_data[i] = (*this->data_)[i] > T{0} ? (*this->data_)[i] : T{0};
//     }
//     AutogradTensor<T> result(this->shape_, result_data, requires_grad_);
//     if (requires_grad_) {
//         result.add_dependency(const_cast<AutogradTensor<T>*>(this));
//         result.set_backward_fn([this_ptr = const_cast<AutogradTensor<T>*>(this),
//                                result_ptr = std::make_shared<AutogradTensor<T>>(result)]() {
//             const auto& result_grad = result_ptr->grad();
//             auto& this_grad = this_ptr->grad();
//             for (size_t i = 0; i < this_ptr->BaseTensor<T>::compute_size(); ++i) {
//                 this_grad[i] += (*this_ptr->data_)[i] > T{0} ? result_grad[i] : T{0};
//             }
//         });
//     }
//     return result;
// }

// // template <typename T>
// // AutogradTensor<T> AutogradTensor<T>::softmax() const {
// //     std::vector<T> exp_data(this->BaseTensor<T>::compute_size());
// //     T max_val = *std::max_element(this->data_->begin(), this->data_->end());
// //     for (size_t i = 0; i < this->BaseTensor<T>::compute_size(); ++i) {
// //         exp_data[i] = std::exp((*this->data_)[i] - max_val);
// //     }
// //     T sum_exp = std::accumulate(exp_data.begin(), exp_data.end(), static_cast<T>(0));
// //     std::vector<T> result_data(this->BaseTensor<T>::compute_size());
// //     for (size_t i = 0; i < this->BaseTensor<T>::compute_size(); ++i) {
// //         result_data[i] = exp_data[i] / sum_exp;
// //     }
// //     AutogradTensor<T> result(this->shape_, result_data, this->requires_grad_);
// //     if (result.requires_grad_) {
// //         result.add_dependency(const_cast<AutogradTensor<T>*>(this));
// //         result.set_backward_fn([this_ptr = const_cast<AutogradTensor<T>*>(this),
// //                                result_ptr = &result, // Capture the result tensor
// //                                result_data = result.data_]() {
// //             const auto& result_grad = result_ptr->grad();
// //             auto& this_grad = this_ptr->grad();
// //             const auto& softmax_output = *result_data;
// //             for (size_t i = 0; i < this_ptr->compute_size(); ++i) {
// //                 T grad_val = 0;
// //                 for (size_t j = 0; j < this_ptr->compute_size(); ++j) {
// //                     T delta = (i == j) ? 1.0 : 0.0;
// //                     grad_val += result_grad[j] * softmax_output[i] * (delta - softmax_output[j]);
// //                 }
// //                 this_grad[i] += grad_val;
// //             }
// //         });
// //     }
// //     return result;
// // }

template class AutogradTensor<double>;
template class AutogradTensor<float>;