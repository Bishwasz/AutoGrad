#include "../AutoGrad/AutoGradTensor.h"
#include <functional>
template <typename T>   
class Loss : public AutogradTensor<T> {

    public:
        // Cross-entropy loss
        static AutogradTensor<T> cross_entropy_loss( AutogradTensor<T>& predictions, const AutogradTensor<T>& labels);


};
template <typename T>
AutogradTensor<T> Loss<T>::cross_entropy_loss( AutogradTensor<T>& predictions, const AutogradTensor<T>& labels) {
    // Check shape compatibility
    if (predictions.shape().size() != 2 || labels.shape().size() != 2 || 
        predictions.shape()[0] != labels.shape()[0] || predictions.shape()[1] != labels.shape()[1]) {
        throw std::runtime_error("Shape mismatch for cross-entropy loss");
    }

    const size_t batch_size = predictions.shape()[0]; // N
    const size_t num_classes = predictions.shape()[1]; // C

    // Compute softmax and cross-entropy loss
    std::vector<T> softmax_probs(batch_size * num_classes);
    T loss_sum = 0.0;

    for (size_t n = 0; n < batch_size; ++n) {
        // Compute softmax for sample n
        T max_logit = predictions.at({static_cast<int>(n),1}); // For numerical stability
        for (size_t i = 1; i < num_classes; ++i) {
            max_logit = std::max(max_logit, predictions.at({static_cast<int>(n) ,  static_cast<int>(i)}));
        }

        T exp_sum = 0.0;
        for (size_t i = 0; i < num_classes; ++i) {
            softmax_probs[n * num_classes + i] = 
                std::exp(predictions.at({static_cast<int>(n) , static_cast<int>(i)}) - max_logit);
            exp_sum += softmax_probs[n * num_classes + i];
        }

        for (size_t i = 0; i < num_classes; ++i) {
            softmax_probs[n * num_classes + i] /= exp_sum;
            T label = labels.at({static_cast<int>(n) ,  static_cast<int>(i)});
            if (label > 0) { // Only include terms where label is positive
                loss_sum -= label * std::log(softmax_probs[n * num_classes + i]);
            }
        }
    }
    

    // Average loss over batch
    T avg_loss = loss_sum / static_cast<T>(batch_size);

    // Create result tensor to store loss and build computation graph
    AutogradTensor<T> result({1}, {avg_loss}, predictions.is_grad() || labels.is_grad());

    // Set up backward pass if gradients are required
    if (result.is_grad()) {
  

        // Add dependencies
        result.add_dependency(const_cast<AutogradTensor<T>*>(&predictions));
        // result.add_dependency(const_cast<AutogradTensor<T>*>(&labels));

        // Set backward function
        result.set_backward_fn([&result,&predictions, &labels, softmax_probs, batch_size, num_classes]() {
            auto& result_grad = result.grad(); // Scalar gradient (dL/dloss)
            T grad_scale = result_grad[0] / static_cast<T>(batch_size); // Scale by 1/N

            // Gradient w.r.t. predictions: (softmax - labels) / batch_size
            // std::cout << "No fault yet 1 " << std::endl;
            if (predictions.is_grad()) {
                auto &pred_grad = predictions.grad();
                
                // std::cout<<pred_grad.size() << std::endl;
                // std::cout<<pred_grad.data()->size() << std::endl;
                for (size_t n = 0; n < batch_size; ++n) {
                    for (size_t i = 0; i < num_classes; ++i) {
                        size_t idx = n * num_classes + i;
                        pred_grad[idx] += grad_scale * (softmax_probs[idx] - labels.at({static_cast<int>(n), static_cast<int>(i)}));
                    }
                }
                // std::cout << "No fault yet 2 " << std::endl;
            }


        });
    }

    return result;
}