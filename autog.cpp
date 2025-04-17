#include <vector>
#include <functional>
#include <memory>
#include <iostream>

class Tensor {
private:
    std::vector<int> shape_;                          // Shape remains on stack
    std::unique_ptr<std::vector<float>> data_;        // Data on heap via unique_ptr
    std::unique_ptr<std::vector<float>> grad_;        // Gradient on heap via unique_ptr
    bool requires_grad_;                              // Flag for gradient tracking
    std::vector<std::shared_ptr<Tensor>> dependencies_; // Dependencies for autograd
    std::function<void()> backward_fn_;               // Backward function

public:
    // Constructor: Initialize tensor with shape and optional data
    Tensor(const std::vector<int>& shape, bool requires_grad = false) 
        : shape_(shape), requires_grad_(requires_grad) {
        int size = compute_size();
        data_ = std::make_unique<std::vector<float>>(size, 0.0f); // Allocate on heap
        if (requires_grad_) {
            grad_ = std::make_unique<std::vector<float>>(size, 0.0f); // Allocate grad on heap
        }
    }

    // Constructor: Initialize with shape and data
    Tensor(const std::vector<int>& shape, const std::vector<float>& data, bool requires_grad = false)
        : shape_(shape), requires_grad_(requires_grad) {
        data_ = std::make_unique<std::vector<float>>(data); // Copy data to heap
        if (data_->size() != compute_size()) {
            throw std::runtime_error("Data size does not match shape");
        }
        if (requires_grad_) {
            grad_ = std::make_unique<std::vector<float>>(data_->size(), 0.0f);
        }
    }
    int compute_size() const {
        int size = 1;
        for (int dim : shape_) size *= dim;
        return size;
    }
    Tensor operator+(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Shape mismatch for addition");
        }
    
        Tensor result(shape_, requires_grad_ || other.requires_grad_);
        for (int i = 0; i < compute_size(); ++i) {
            (*result.data_)[i] = (*data_)[i] + (*other.data_)[i];
        }
    
        if (result.requires_grad_) {
            // Store dependencies
            result.dependencies_ = {std::make_shared<Tensor>(*this), std::make_shared<Tensor>(other)};
    
            // Define backward function
            result.backward_fn_ = [this_dep = result.dependencies_[0], other_dep = result.dependencies_[1]]() {
                // Gradient w.r.t. this
                if (this_dep->requires_grad_) {
                    for (int i = 0; i < this_dep->compute_size(); ++i) {
                        (*this_dep->grad_)[i] += 1.0f;
                    }
                }
                // Gradient w.r.t. other
                if (other_dep->requires_grad_) {
                    for (int i = 0; i < other_dep->compute_size(); ++i) {
                        (*other_dep->grad_)[i] += 1.0f;
                    }
                }
            };
        }
    
        return result;
    }
    Tensor operator+(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Shape mismatch for addition");
        }
    
        Tensor result(shape_, requires_grad_ || other.requires_grad_);
        for (int i = 0; i < compute_size(); ++i) {
            (*result.data_)[i] = (*data_)[i] + (*other.data_)[i];
        }
    
        if (result.requires_grad_) {
            // Store dependencies
            result.dependencies_ = {std::make_shared<Tensor>(*this), std::make_shared<Tensor>(other)};
    
            // Define backward function
            result.backward_fn_ = [this_dep = result.dependencies_[0], other_dep = result.dependencies_[1]]() {
                // Gradient w.r.t. this
                if (this_dep->requires_grad_) {
                    for (int i = 0; i < this_dep->compute_size(); ++i) {
                        (*this_dep->grad_)[i] += 1.0f;
                    }
                }
                // Gradient w.r.t. other
                if (other_dep->requires_grad_) {
                    for (int i = 0; i < other_dep->compute_size(); ++i) {
                        (*other_dep->grad_)[i] += 1.0f;
                    }
                }
            };
        }
    
        return result;
    }
}