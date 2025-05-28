#ifndef AUTOGRAD_TENSOR_H
#define AUTOGRAD_TENSOR_H
#include "../BaseTensor/BaseTensor.h"
#include <functional>

template <typename T>   
class AutogradTensor : public BaseTensor<T> {
    public:
        AutogradTensor() : BaseTensor<T>({1}), requires_grad_(false) {
            grad_ = nullptr; // Explicitly initialize grad_ to nullptr
        }
        AutogradTensor(const std::vector<int>& shape, const std::vector<T>& data, bool requires_grad = false);
        AutogradTensor(const std::vector<int>& shape, bool requires_grad = false);

        // Getters
        std::vector<T>& grad() { return *grad_; }
        const std::vector<T>& grad() const { return *grad_; }
        bool requires_grad() const;

        virtual T& at(const std::vector<int>& indices);
        virtual const T& at(const std::vector<int>& indices) const;

        AutogradTensor(AutogradTensor&& other) noexcept = default;
        AutogradTensor& operator=(AutogradTensor&& other) noexcept = default;
        AutogradTensor& operator=(const AutogradTensor& other) = default;
            // Setters/Modifiers a nd Modifiers
        void set_requires_grad(bool req_grad);
        void zero_grad();
        void one_grad(); 

        // Autograd
        void add_dependency(AutogradTensor<T> *dep);
        void set_backward_fn(std::function<void()> fn);
        void backward();

        // Operations (override to include gradient computation)
        AutogradTensor<T> operator+( AutogradTensor<T>& other)  ;
        AutogradTensor<T> operator-(AutogradTensor<T>& other);
        AutogradTensor<T> operator*( AutogradTensor<T>& other)  ;
        AutogradTensor<T> log();
        AutogradTensor<T> relu();
        AutogradTensor<T> broadcast_add(AutogradTensor<T>& other) ;
        AutogradTensor<T> softmax();
        AutogradTensor<T> cross_entropy_loss(AutogradTensor<T>& labels);
        
        // AutogradTensor<T> cross_entropy_loss(AutogradTensor<T>& other) ;


    protected:
        std::unique_ptr<std::vector<T>> grad_;
        bool requires_grad_;
        std::vector<AutogradTensor<T>*> dependencies_;
        std::function<void()> backward_fn_;
};
#include "AutoGradTensor.tpp"
#endif