#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>       // For shared_ptr, unique_ptr, weak_ptr, enable_shared_from_this
#include <functional>   // For std::function
#include <string>       // Often needed indirectly or for potential future methods

// Forward declaration (optional, can sometimes help reduce header dependencies)
// class Tensor; // Not strictly needed here as we define it fully

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    std::vector<int> shape_;
    std::unique_ptr<std::vector<float>> data_;
    std::unique_ptr<std::vector<float>> grad_;
    bool requires_grad_;
    std::vector<std::shared_ptr<Tensor>> dependencies_;
    std::function<void()> backward_fn_;

public:
    // --- Constructors ---
    Tensor(const std::vector<int>& shape, const std::vector<float>& data, bool requires_grad = true);
    Tensor(const std::vector<int>& shape, bool requires_grad = true);
    // --- Getters ---
    size_t compute_size() const;
    const std::vector<float>& data() const;
    const std::vector<float>& grad() const; // Const getter
    std::vector<float>& data(); // Non-const getter for data
    std::vector<float>& grad(); // Non-const getter
    const std::vector<int>& shape() const;
    bool requires_grad() const;

    // --- Setters / Modifiers ---
     void set_requires_grad(bool req_grad);
     void zero_grad();
     void one_grad();
    void set_data(const std::vector<float>& data);

    void add_dependency(std::shared_ptr<Tensor> dep);
    void set_backward_fn(std::function<void()> fn);
    // --- Autograd ---
    void backward();

    // --- Operators ---
    // Note: Operators are declared as member functions returning shared_ptr
    std::shared_ptr<Tensor> operator+(const Tensor& other) const;
    std::shared_ptr<Tensor> operator*(const Tensor& other) const; // Matrix multiplication
    std::shared_ptr<Tensor> broadcast_add(const Tensor& other) const;
    std::shared_ptr<Tensor> Tensor::log() const;
    std::shared_ptr<Tensor> Tensor::neg() const;

    // Potentially add other operators here (subtraction, element-wise *, etc.)

}; // End of Tensor class declaration

#endif // TENSOR_H