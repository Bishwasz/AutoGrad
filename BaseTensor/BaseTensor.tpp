#ifndef BASE_TENSOR_TPP
#define BASE_TENSOR_TPP

#include "BaseTensor.h"
#include <random>
#include "Mul.cpp"
#include <cblas.h>

template <typename T>
BaseTensor<T>::BaseTensor(const std::vector<int>& shape, const std::vector<T>& data)
    : shape_(shape){
    // size_t size = compute_size();
    // if (data.size() != size) {
    //     std::string shape_str;
    //     for (size_t i = 0; i < shape_.size(); ++i) {
    //         shape_str += std::to_string(shape_[i]) + (i == shape_.size()  ? "" : ", ");
    //     }
    //     throw std::runtime_error("Data size (" + std::to_string(data.size()) + ") does muji  not match shape [" + shape_str + "] which requires size " + std::to_string(size));
    // }
    data_ = std::make_unique<std::vector<T>>(data);

}
// template <typename T>
// BaseTensor<T>& BaseTensor<T>::operator=(const BaseTensor<T>& other) {
//     if (this != &other) {
//         shape_ = other.shape_;
//         data_ = std::make_unique<std::vector<T>>(*other.data_);
//     }
//     return *this;
// }

template <typename T>
BaseTensor<T>::BaseTensor(const std::vector<int>& shape)
    : shape_(shape) {
    for (int dim : shape_) {
        if (dim < 0) {
            throw std::runtime_error("Negative dimension in shape");
        }
    }

    size_t size = compute_size();
    data_ = std::make_unique<std::vector<T>>(size);

    std::random_device rd;
    std::mt19937 gen(rd());

    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(0, 100);  // Adjust range as needed
        for (size_t i = 0; i < size; ++i) {
            (*data_)[i] = dist(gen);
        }
    } else if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(0.0, 1.0);
        for (size_t i = 0; i < size; ++i) {
            (*data_)[i] = dist(gen);
        }
    } else {
        throw std::runtime_error("Random initialization not supported for this type");
    }
}

template <typename T>
T& BaseTensor<T>::at(const std::vector<int>& indices) {
    if (indices.size() != shape_.size())
        throw std::invalid_argument("Incorrect number of indices");

    size_t offset = 0;
    size_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape_[i])
            throw std::out_of_range("Index out of bounds");

        offset += indices[i] * stride;
        stride *= shape_[i];
    }

    return (*data_)[offset];
}

template <typename T>
const T& BaseTensor<T>::at(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size())
        throw std::invalid_argument("Incorrect number of indices");

    size_t offset = 0;
    size_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape_[i])
            throw std::out_of_range("Index out of bounds");

        offset += indices[i] * stride;
        stride *= shape_[i];
    }

    return (*data_)[offset];
}

template <typename T>
void BaseTensor<T>::set_data(const std::vector<T>& data) {
    if (data.size() != compute_size()) {
        throw std::invalid_argument("Data size does not match tensor shape");
    }
    *data_ = data;
}

// getter
template <typename T>
const std::vector<int>& BaseTensor<T>::shape() const {
    return shape_;
}
template <typename T>   
const std::vector<T>& BaseTensor<T>::data() const {
    if (!data_) {
        throw std::runtime_error("Tensor data is null");
    }
    return *data_;
}
template <typename T>
size_t BaseTensor<T>::compute_size() const {
    size_t size = 1;
    for (int dim : shape_) {
        size *= dim;
    }
    return size;
}


template <typename T>
BaseTensor<T> BaseTensor<T>::operator+(const BaseTensor<T>& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for addition");
    }
    auto result =BaseTensor<T>(shape_);
    size_t size = compute_size();
    for (size_t i = 0; i < size; ++i) {
        (*result.data_)[i] = (*data_)[i] + (*other.data_)[i];
    }
    return result;
}

// naive implementation of matrix multiplication

void print_tensor(const std::vector<float>& tensor) {
    std::cout << "Tensor data: ";
    for (const auto& val : tensor) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
template <typename T>
BaseTensor<T> BaseTensor<T>::operator*(const BaseTensor<T>& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::runtime_error("Matrix multiplication requires 2D tensors");
    }
    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t N = other.shape_[1];
    if (K != static_cast<size_t>(other.shape_[0])) {
        throw std::runtime_error("Incompatible dimensions for matrix multiplication");
    }

    std::vector<int> result_shape = {static_cast<int>(M), static_cast<int>(N)};
    auto result = BaseTensor<T>(result_shape);
    matrixMultiply(this->data(), other.data(), result.data(), M, K, N);

    // openBlasMultiply(this->data(), other.data(), result.data(), M, K, N);

      
    return result;
}
template <typename T>
BaseTensor<T> BaseTensor<T>::broadcast_add(const BaseTensor<T>& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::runtime_error("Broadcast addition requires 2D tensors");
    }
    size_t M = shape_[0];
    size_t N = shape_[1];
    if (static_cast<size_t>(other.shape_[0]) != 1 || static_cast<size_t>(other.shape_[1]) ){
        throw std::runtime_error("Incompatible shapes for broadcast addition");
    }

    auto result = BaseTensor<T>(shape_);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            (*result.data_)[i * N + j] = (*data_)[i * N + j] + (*other.data_)[j];
        }
    }
    return result;
}

template <typename T>
std::vector<T>& BaseTensor<T>::data() {
    if (!data_) {
        throw std::runtime_error("Tensor data is null");
    }
    return *data_;
}

template <typename T>
BaseTensor<T>BaseTensor<T>::log() const {
    auto result = BaseTensor<T>(shape_);
    size_t size = compute_size();
    for (size_t i = 0; i < size; ++i) {
        (*result.data_)[i] = std::log((*data_)[i]);
    }
    return result;
}

template <typename T>
BaseTensor<T>BaseTensor<T>::neg() const {
    auto result = BaseTensor<T>(shape_);
    size_t size = compute_size();
    for (size_t i = 0; i < size; ++i) {
        (*result.data_)[i] = -(*data_)[i];
    }
    return result;
}
template <typename T>
BaseTensor<T>BaseTensor<T>::operator*(T scalar){
    auto result = BaseTensor<T>(shape_);
    size_t size = compute_size();
    for (size_t i = 0; i < size; ++i) {
        (*result.data_)[i] = (*data_)[i] * scalar;
    }
    return result;
}
template<typename T>
BaseTensor<T> operator*(T scalar, const BaseTensor<T>& tensor) {
    BaseTensor<T> result(tensor.shape());
    for (size_t i = 0; i < result.data().size(); ++i) {
        result.data()[i] = scalar * tensor.data()[i];
    }
    return result;
}
// In your BaseTensor.h or BaseTensor.tpp file

template <typename T>
BaseTensor<T>& BaseTensor<T>::operator-=(const BaseTensor<T>& other) {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for in-place subtraction (-=)");
    }

    size_t size = compute_size();
    for (size_t i = 0; i < size; ++i) {
        (*data_)[i] -= (*other.data_)[i];
    }

    // Return a reference to the modified object
    return *this;
}


#endif // BASE_TENSOR_TPP
