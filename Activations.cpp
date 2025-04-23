#include <cmath>
#include <memory>
#include <vector>
#include <tensor.h>
#include <activations.h>

std::shared_ptr<Tensor> sigmoid(const Tensor& input) {
    // Create result tensor with same shape and requires_grad
    auto result = std::make_shared<Tensor>(input.shape(), input.requires_grad());

    // Forward pass: compute sigmoid
    const auto& input_data = input.data();
    auto& result_data = result->data(); // Use non-const data() getter
    size_t size = input.compute_size();
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
    }

    // Backward pass: set up autograd
    if (result->requires_grad()) {
        auto input_shared = std::const_pointer_cast<Tensor>(input.shared_from_this());
        result->add_dependency(input_shared);

        result->set_backward_fn([result_wptr = std::weak_ptr<Tensor>(result),
                                input_wptr = std::weak_ptr<Tensor>(input_shared)]() {
            auto result_locked = result_wptr.lock();
            auto input_tensor = input_wptr.lock();
            if (!result_locked || !input_tensor) return;

            if (input_tensor->requires_grad()) {
                const auto& result_data = result_locked->data();
                const auto& result_grad = result_locked->grad();
                auto& input_grad = input_tensor->grad();
                size_t size = input_tensor->compute_size();

                // Gradient of sigmoid: grad_input = grad_output * sigmoid(x) * (1 - sigmoid(x))
                for (size_t i = 0; i < size; ++i) {
                    float sigmoid_val = result_data[i];
                    float grad = result_grad[i] * sigmoid_val * (1.0f - sigmoid_val);
                    if (i < input_grad.size()) {
                        input_grad[i] += grad;
                    }
                }
            }
        });
    }

    return result;
}
std::shared_ptr<Tensor> relu(const Tensor& input) {
    // Create result tensor with same shape and requires_grad
    auto result = std::make_shared<Tensor>(input.shape(), input.requires_grad());

    // Forward pass: compute sigmoid
    const auto& input_data = input.data();
    auto& result_data = result->data(); // Use non-const data() getter
    size_t size = input.compute_size();
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = std::max(0.0f, input_data[i]);
    }

    // Backward pass: set up autograd
    if (result->requires_grad()) {
        auto input_shared = std::const_pointer_cast<Tensor>(input.shared_from_this());
        result->add_dependency(input_shared);

        result->set_backward_fn([result_wptr = std::weak_ptr<Tensor>(result),
                                input_wptr = std::weak_ptr<Tensor>(input_shared)]() {
            auto result_locked = result_wptr.lock();
            auto input_tensor = input_wptr.lock();
            if (!result_locked || !input_tensor) return;

            if (input_tensor->requires_grad()) {
                const auto& result_data = result_locked->data();
                const auto& result_grad = result_locked->grad();
                auto& input_grad = input_tensor->grad();
                size_t size = input_tensor->compute_size();

                // Gradient of sigmoid: grad_input = grad_output * sigmoid(x) * (1 - sigmoid(x))
                for (size_t i = 0; i < size; ++i) {
                    float sigmoid_val = result_data[i];
                    float grad = result_grad[i] * (sigmoid_val > 0 ? 1.0f : 0.0f);
                    if (i < input_grad.size()) {
                        input_grad[i] += grad;
                    }
                }
            }
        });
    }

    return result;
}

