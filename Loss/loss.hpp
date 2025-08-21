#include "../AutoGrad/AutoGradTensor.h"
#include <functional>
#include <numeric> // For std::max_element if needed

// DESIGN: Removed unnecessary inheritance from AutogradTensor<T>.
// This class is now just a container for static loss functions.
template <typename T>
class Loss {
public:
    static AutogradTensor<T> cross_entropy_loss(const AutogradTensor<T>& predictions, const AutogradTensor<T>& labels);
};


template <typename T>
AutogradTensor<T> Loss<T>::cross_entropy_loss(const AutogradTensor<T>& predictions, const AutogradTensor<T>& labels) {
    if (predictions.shape() != labels.shape()) {
        throw std::runtime_error("Shape mismatch for cross-entropy loss");
    }

    const size_t batch_size = predictions.shape()[0];
    const size_t num_classes = predictions.shape()[1];

    std::vector<T> softmax_probs(batch_size * num_classes);
    T loss_sum = 0.0;

    // --- Forward Pass (This part was already well-implemented) ---
    for (size_t n = 0; n < batch_size; ++n) {
        // Find max logit for numerical stability (log-sum-exp trick)
        T max_logit = predictions.at({(int)n, 0});
        for (size_t i = 1; i < num_classes; ++i) {
            max_logit = std::max(max_logit, predictions.at({(int)n, (int)i}));
        }

        T exp_sum = 0.0;
        std::vector<T> exps(num_classes);
        for (size_t i = 0; i < num_classes; ++i) {
            exps[i] = std::exp(predictions.at({(int)n, (int)i}) - max_logit);
            exp_sum += exps[i];
        }

        for (size_t i = 0; i < num_classes; ++i) {
            size_t idx = n * num_classes + i;
            softmax_probs[idx] = exps[i] / exp_sum;
            
            T label = labels.at({(int)n, (int)i});
            if (label > 0) { // Assumes one-hot or soft labels
                const T epsilon = 1e-12; // Prevent log(0)
                loss_sum -= label * std::log(std::max(softmax_probs[idx], epsilon));
            }
        }
    }

    T avg_loss = loss_sum / static_cast<T>(batch_size);

    // Create result tensor for the scalar loss
    AutogradTensor<T> result({1}, {avg_loss}, predictions.is_grad());

    // --- Backward Pass ---
    if (result.is_grad()) {
        // The backward pass needs the softmax probabilities from the forward pass.
        // Capturing `softmax_probs` by value is correct and necessary.
        result.add_dependency(const_cast<AutogradTensor<T>*>(&predictions));

        // CORRECTED: The lambda now accepts a `self` pointer to avoid the dangling reference.
        result.set_backward_fn([&predictions, &labels, softmax_probs, batch_size, num_classes]
                               (AutogradTensor<T>* self) {
            // `self` is the pointer to the `result` tensor. No more dangling reference!
            T grad_of_loss = self->grad().at({0}); // Gradient of the final output (usually 1.0)

            if (predictions.is_grad()) {
                auto& pred_grad = const_cast<AutogradTensor<T>&>(predictions).grad().data();
                T grad_scale = grad_of_loss / static_cast<T>(batch_size);

                // Gradient is (softmax - labels) / N
                for (size_t idx = 0; idx < batch_size * num_classes; ++idx) {
                    // This direct indexing is possible because shapes are guaranteed to match.
                    pred_grad[idx] += grad_scale * (softmax_probs[idx] - labels.data()[idx]);
                }
            }
        });
    }

    return result;
}