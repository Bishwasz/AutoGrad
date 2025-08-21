#ifndef AUTOGRAD_TENSOR_H
#define AUTOGRAD_TENSOR_H

#include "../BaseTensor/BaseTensor.h"
#include <functional>

// Forward declaration for the class itself, useful for the using alias
template <typename T>
class AutogradTensor;

// It's good practice to create a type alias for the backward function signature
template<typename T>
using BackwardFn = std::function<void(AutogradTensor<T>*)>;


template <typename T>
class AutogradTensor : public BaseTensor<T> {
    public:
        AutogradTensor() : BaseTensor<T>({1}), requires_grad_(false) {
        grad_ = BaseTensor<T>({1});
        grad_.zero_data();
        }
        AutogradTensor(const std::vector<int>& shape, const std::vector<T>& data, bool requires_grad = false);
        AutogradTensor(const std::vector<int>& shape, bool requires_grad = false);

        // Getters
        BaseTensor<T>& grad() { return grad_; }
        const std::vector<int>& shape() const { return this->shape_; }
        bool requires_grad() const;

        virtual T& at(const std::vector<int>& indices);

        virtual const T& at(const std::vector<int>& indices) const;
        virtual const T& at_grad(const std::vector<int>& indices) const;

        AutogradTensor(AutogradTensor&& other) noexcept = default;
        AutogradTensor& operator=(AutogradTensor&& other) noexcept = default;
        AutogradTensor& operator=(const AutogradTensor& other) = default;

        // Setters/Modifiers
        void set_requires_grad(bool req_grad);
        void zero_grad();
        void one_grad();
        bool is_grad() const { return requires_grad_; }

        // Autograd
        void add_dependency(AutogradTensor<T> *dep);
        // CORRECTED #1: Update the backward function signature to prevent memory errors
        void set_backward_fn(BackwardFn<T> fn);
        void backward();

        // Operations (override to include gradient computation)
        // CORRECTED #2: Add `const` to match the implementation file and fix compiler errors
        AutogradTensor<T> operator+( const AutogradTensor<T>& other);
        AutogradTensor<T> operator-( const AutogradTensor<T>& other);
        AutogradTensor<T> operator*( const AutogradTensor<T>& other);
        AutogradTensor<T> operator*(T scalar);
        AutogradTensor<T> log();
        AutogradTensor<T> relu();
        AutogradTensor<T> broadcast_add( const AutogradTensor<T>& other);
        AutogradTensor<T> softmax();

        // Operation for BaseTensor
        AutogradTensor<T> operator+(const BaseTensor<T>& other);
        AutogradTensor<T> operator*(BaseTensor<T>& other);
        AutogradTensor<T> operator-(const BaseTensor<T>& other);
        
    protected:
        BaseTensor<T> grad_;
        std::vector<int> grad_shape_;
        bool requires_grad_;
        std::vector<AutogradTensor<T>*> dependencies_;
        // CORRECTED #1: Update the member variable to the new function signature
        BackwardFn<T> backward_fn_;
};

#include "AutoGradTensor.tpp"
#include "get_set.tpp"
#include "init.tpp"
#include "BaseTensorOperator.tpp"

#endif // AUTOGRAD_TENSOR_H