
#include <random>
#include <functional>
#include <memory>
#include <unordered_set>
#include <algorithm> 

template <typename T>
AutogradTensor<T> AutogradTensor<T>::operator+( const BaseTensor<T>& other){
    if (this->shape_ != other.shape()) {
        throw std::runtime_error("Shape mismatch for addition");
    }
    std::vector<T> result_data(this->compute_size());
    for (size_t i = 0; i <this->compute_size(); ++i) {
        result_data[i] = (*this->data_)[i] + other.data()[i];
    }
    AutogradTensor<T> result(this->shape_, result_data, this->requires_grad_);
    return result;
 }
 template <typename T>
AutogradTensor<T> AutogradTensor<T>::operator-( const BaseTensor<T>& other){
    if (this->shape_ != other.shape()) {
        throw std::runtime_error("Shape mismatch for addition");
    }
    std::vector<T> result_data(this->compute_size());
    for (size_t i = 0; i <this->compute_size(); ++i) {
        result_data[i] = (*this->data_)[i] - other.data()[i];
    }
    AutogradTensor<T> result(this->shape_, result_data, this->requires_grad_);
    return result;


 }
template <typename T>
 AutogradTensor<T> AutogradTensor<T>::operator*(T scalar){
    std::vector<T> result_data(this->compute_size(),true);
    for (size_t i = 0; i < this->compute_size(); ++i) {
        result_data[i] = (*this->data_)[i] * scalar;
    }
    AutogradTensor<T> result(this->shape_, result_data, this->requires_grad_);
    return result;
 }
template<typename T>
AutogradTensor<T> operator*(T scalar, const AutogradTensor<T>& tensor) {
    AutogradTensor<T> result(tensor.shape(),true);
    for (size_t i = 0; i < result.data().size(); ++i) {
        result.data()[i] = scalar * tensor.data()[i];
    }
    return result;
}