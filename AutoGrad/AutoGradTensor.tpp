
#include <random>
#include <functional>
#include <memory>
#include <unordered_set>
#include <algorithm> // for std::max_element


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
        if (!node) std::cerr << "Warning: Null node in backward pass" << std::endl;
        if(!node->backward_fn_) std::cerr << "Warning: No backward function set for node" << std::endl;

        if (node && node->backward_fn_) {
            node->backward_fn_();
        }
    }
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
AutogradTensor<T> AutogradTensor<T>::operator*(AutogradTensor<T>& other) {
    if (this->shape_[1] != other.shape()[0]) {
        throw std::runtime_error("Shape mismatch for multiplication: incompatible inner dimensions.");
    }

    int M = this->shape_[0];
    int K = this->shape_[1];
    int N = other.shape()[1];  // ✅ Corrected

    std::vector<T> result_data(M * N, T(0));

    // Matrix multiplication
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += (*this->data_)[i * K + k] * other.data()[k * N + j];
            }
            result_data[i * N + j] = sum;
        }
    }

    // Correct output shape
    AutogradTensor<T> result({M, N}, result_data, requires_grad_ || other.requires_grad_);

    if (result.requires_grad_) {
        result.add_dependency(this);
        result.add_dependency(&other);

        result.set_backward_fn([this, other_ptr = &other, M, K, N, &result]() {
            const auto& result_grad = result.grad();
            // std::cout<<"No fault yet 3 " << std::endl;


            if (this->requires_grad()) {
                auto& this_grad = this->grad();
                for (int i = 0; i < M; ++i) {
                    for (int k = 0; k < K; ++k) {
                        T grad_val = 0;
                        for (int j = 0; j < N; ++j) {
                            grad_val += result_grad[i * N + j] * other_ptr->data()[k * N + j];
                        }
                        this_grad[i * K + k] += grad_val;
                    }
                }
            }

            if (other_ptr->requires_grad()) {
                auto& other_grad = other_ptr->grad();
                // std::cout<<"No fault yet 4 " << std::endl;
                for (int k = 0; k < K; ++k) {
                    for (int j = 0; j < N; ++j) {
                        T grad_val = 0;
                        for (int i = 0; i < M; ++i) {
                            grad_val += (*this->data_)[i * K + k] * result_grad[i * N + j];
                        }
                        other_grad[k * N + j] += grad_val;
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