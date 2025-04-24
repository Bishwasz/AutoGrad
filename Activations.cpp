#include <cmath>
#include <memory>
#include <vector>
#include <tensor.h>
#include <activations.h>
#include <stdexcept>

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

std::shared_ptr<Tensor> softmax(const Tensor& input) {
    // Create result tensor with same shape and requires_grad
    auto result = std::make_shared<Tensor>(input.shape(), input.requires_grad());
    
    // Ensure input is at least 2D
    const auto& shape = input.shape();
    if (shape.size() < 2) {
        throw std::runtime_error("Softmax requires at least 2D input tensor");
    }
    
    size_t num_rows = shape[0];
    size_t num_cols = shape[1];

    // Forward pass: compute softmax row-wise
    const auto& input_data = input.data();
    auto& result_data = result->data();
    size_t offset = 0;

    for (size_t row = 0; row < num_rows; ++row) {
        // Find max value in the row for numerical stability
        float max_val = input_data[offset];
        for (size_t col = 1; col < num_cols; ++col) {
            max_val = std::max(max_val, input_data[offset + col]);
        }

        // Compute exponentials and sum for the row
        float sum_exp = 0.0f;
        for (size_t col = 0; col < num_cols; ++col) {
            result_data[offset + col] = std::exp(input_data[offset + col] - max_val);
            sum_exp += result_data[offset + col];
        }

        // Normalize to get softmax probabilities for the row
        for (size_t col = 0; col < num_cols; ++col) {
            result_data[offset + col] /= sum_exp;
        }

        offset += num_cols;
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
                const auto& shape = input_tensor->shape();
                size_t num_rows = shape[0];
                size_t num_cols = shape[1];
                size_t offset = 0;

                // Gradient of softmax row-wise
                for (size_t row = 0; row < num_rows; ++row) {
                    for (size_t i = 0; i < num_cols; ++i) {
                        float grad = 0.0f;
                        for (size_t j = 0; j < num_cols; ++j) {
                            float delta_ij = (i == j) ? 1.0f : 0.0f;
                            grad += result_grad[offset + j] * result_data[offset + i] * (delta_ij - result_data[offset + j]);
                        }
                        if (offset + i < input_grad.size()) {
                            input_grad[offset + i] += grad;
                        }
                    }
                    offset += num_cols;
                }
            }
        });
    }

    return result;
}

std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> predictions, std::shared_ptr<Tensor> labels) {
    // Ensure predictions and labels have the same shape
    if (predictions->shape() != labels->shape()) {
        throw std::runtime_error("Predictions and labels must have the same shape");
    }

    // Get data from tensors
    const auto& pred_data = predictions->data();
    const auto& label_data = labels->data();
    if (pred_data.size() != label_data.size()) {
        throw std::runtime_error("Data size mismatch between predictions and labels");
    }

    // Step 1: Compute log(predictions) element-wise
    std::vector<float> log_preds(pred_data.size());
    for (size_t i = 0; i < pred_data.size(); ++i) {
        if (pred_data[i] <= 0.0f) {
            // Handle non-positive values to avoid log(0) or log(negative)
            log_preds[i] = std::log(1e-15f); // Small constant for numerical stability
        } else {
            log_preds[i] = std::log(pred_data[i]);
        }
    }

    // Step 2: Compute labels * log(predictions) element-wise
    std::vector<float> weighted_log_preds(pred_data.size());
    for (size_t i = 0; i < pred_data.size(); ++i) {
        weighted_log_preds[i] = label_data[i] * log_preds[i];
    }

    // Step 3: Negate element-wise
    std::vector<float> neg_log_likelihood(pred_data.size());
    for (size_t i = 0; i < pred_data.size(); ++i) {
        neg_log_likelihood[i] = -weighted_log_preds[i];
    }

    // Step 4: Compute mean of all elements
    float sum = 0.0f;
    for (const auto& val : neg_log_likelihood) {
        sum += val;
    }
    float mean = sum / neg_log_likelihood.size();

    // Create a scalar tensor for the loss
    auto loss_tensor = std::make_shared<Tensor>(std::vector<int>{1}, std::vector<float>{mean}, false);

    // Note: For autograd, you need to ensure gradients are tracked.
    // Assuming Tensor supports operator overloading and backward(),
    // we can use the original operations for gradient computation.
    auto log_preds_tensor = std::make_shared<Tensor>(predictions->shape(), log_preds, true);
    auto weighted_log_preds_tensor = std::make_shared<Tensor>(predictions->shape(), weighted_log_preds, true);
    auto neg_log_likelihood_tensor = std::make_shared<Tensor>(predictions->shape(), neg_log_likelihood, true);

    // Recompute the operations using Tensor operations to build the computational graph
    auto tensor_log_preds = predictions->log(); // Should match log_preds
    auto tensor_weighted = *labels * *tensor_log_preds;
    auto tensor_neg = tensor_weighted.neg();
    auto tensor_mean = tensor_neg->mean();

    return tensor_mean; // Return the tensor with the computational graph
}
