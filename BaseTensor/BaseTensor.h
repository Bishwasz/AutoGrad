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

    BaseTensor(BaseTensor&& other) noexcept = default;
    BaseTensor& operator=(BaseTensor&& other) noexcept = default;
    BaseTensor& operator=(const BaseTensor& other);

    // Getters
    virtual const std::vector<int>& shape() const;
    virtual const std::vector<T>& data() const;
    virtual std::vector<T>& data();

    virtual T& at(const std::vector<int>& indices);
    virtual const T& at(const std::vector<int>& indices) const;

    // Setters/Modifiers
    virtual void set_data(const std::vector<T>& data);

    // Operations
     BaseTensor<T> operator+(const BaseTensor<T>& other) const;
     BaseTensor<T> operator*(const BaseTensor<T>& other) const;
     BaseTensor<T> broadcast_add(const BaseTensor<T>& other) const;
     BaseTensor<T> log() const;
     BaseTensor<T> neg() const;

protected:
    std::vector<int> shape_;
    std::unique_ptr<std::vector<T>> data_;

    size_t compute_size() const;
};

#include "BaseTensor.tpp"

#endif // BASE_TENSOR_H