// activations.h
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <memory>

class Tensor;  // Forward declaration

std::shared_ptr<Tensor> sigmoid(const Tensor& input);
std::shared_ptr<Tensor> relu(const Tensor& input);

#endif
