#ifndef BASE_TENSOR_H
#define BASE_TENSOR_H

#include <vector>
#include <memory>
#include <stdexcept>

template <typename T>
class BaseTensor {
public:
    // Constructors
    BaseTensor() : shape_({1}), data_(std::make_unique<std::vector<T>>(1, T{0})) {}
    BaseTensor(const std::vector<int>& shape, const std::vector<T>& data);
    BaseTensor(const std::vector<int>& shape); // Zero initialization
    virtual ~BaseTensor() = default;

    BaseTensor(const BaseTensor<T>& other): shape_(other.shape_), data_(std::make_unique<std::vector<T>>(*other.data_)) {}
    BaseTensor& operator=(const BaseTensor<T>& other) {
    if (this != &other) {
        shape_ = other.shape_;
        data_ = std::make_unique<std::vector<T>>(*other.data_);
    }
    return *this;
}
    BaseTensor(BaseTensor<T>&& other) noexcept
    : shape_(std::move(other.shape_)), data_(std::move(other.data_)) {}
    BaseTensor& operator=(BaseTensor<T>&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        data_ = std::move(other.data_);
    }
    return *this;
}

    // Getters
    virtual const std::vector<int>& shape() const;
    virtual const std::vector<T>& data() const;
    virtual std::vector<T>& data();

    virtual T& at(const std::vector<int>& indices);
    virtual const T& at(const std::vector<int>& indices) const;

    // Setters/Modifiers
    virtual void set_data(const std::vector<T>& data);
    virtual void zero_data() { std::fill(data_->begin(), data_->end(), T{0}); }
    virtual void one_data() { std::fill(data_->begin(), data_->end(), T{1}); }

    // Operations
     BaseTensor<T> operator+(const BaseTensor<T>& other) const;
     BaseTensor<T> operator*(const BaseTensor<T>& other) const;
     BaseTensor<T> broadcast_add(const BaseTensor<T>& other) const;
     BaseTensor<T> log() const;
     BaseTensor<T> neg() const;
     BaseTensor<T> operator*(T scalar);
     BaseTensor<T>& operator-=(const BaseTensor<T>& other);

protected:
    std::vector<int> shape_;
    std::unique_ptr<std::vector<T>> data_;

    size_t compute_size() const;
};

#include "BaseTensor.tpp"

#endif // BASE_TENSOR_H