#include "tensor.h" // Assuming the files are saved as tensor.h and tensor.cpp
#include <iostream>
#include <vector>
#include <activations.h>
#include <fstream>
#include <sstream>
#include <cmath>


void print_tensor(const std::vector<float> c) {

    for (const auto& val : c) {
        std::cout << val << " ";
    }
    std::cout << "\n";
}
// One-hot encode a label (0-9) into a vector of size 10
std::vector<float> one_hot(int label, int num_classes = 10) {
    std::vector<float> encoded(num_classes, 0.0f);
    encoded[label] = 1.0f;
    return encoded;
}

// Cross-entropy loss (assumes predictions are softmax outputs)
std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> predictions, std::shared_ptr<Tensor> labels) {
    // Simplified cross-entropy: -sum(labels * log(predictions))
    auto log_preds = predictions->log();
    auto neg_log_likelihood = (*labels * *log_preds).neg();
    return neg_log_likelihood->mean(); // Scalar loss
}

// SGD update rule
void sgd_update(std::shared_ptr<Tensor> param, float learning_rate) {
    auto grad = param->grad();
    auto data = param->data();
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= learning_rate * grad[i];
    }
    param->set_data(data); // Update tensor data
    param->zero_grad();    // Reset gradients
}

int main() {
    std::vector<int> shape = {2, 5};
    std::vector<float> data= {1, 2 , 3, 4, 1, 
                              6, 7 , 8, 9, 1};
    std::shared_ptr<Tensor> B = std::make_shared<Tensor>(shape, data,true);
    auto soft=softmax(*B);
    soft->backward();
    print_tensor(soft->data());
    print_tensor(B->grad());

    int n = 1; // Number of lines you want to read
    int lines_read = 1;
    int batch_size = 30;

    std::ifstream file("mnist_train.csv");
    if (!file.is_open()) {
        std::cerr << "Failed to open file.\n";
        return 1;
    }

    std::string line;
    std::vector<float> all_data;
    std::vector<float> all_labels;
    int sample_count = 0;

    auto data_batch = std::make_shared<Tensor>(std::vector<int>{batch_size, 784}, false);
    auto label_batch = std::make_shared<Tensor>(std::vector<int>{batch_size, 1}, false);

    auto first_weight= std::make_shared<Tensor>(std::vector<int>{784, 20}, true);
    auto first_bias= std::make_shared<Tensor>(std::vector<int>{1,20}, true);

    auto second_weight= std::make_shared<Tensor>(std::vector<int>{20, 10}, true);
    auto second_bias= std::make_shared<Tensor>(std::vector<int>{1,10}, true);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
    
        // Get the label
        std::getline(ss, cell, ',');
        float label = std::stof(cell);
        all_labels.push_back(label);
    
        // Get the 784 features
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stof(cell));
        }
    
        if (row.size() != 784) {
            std::cerr << "Skipping malformed line " << lines_read + 1 << "\n";
            continue;
        }
    
        all_data.insert(all_data.end(), row.begin(), row.end());
       
        if (lines_read % batch_size == 0) {
            data_batch->set_data(all_data);
            label_batch->set_data(all_labels);
            print_tensor(label_batch->data());
            all_data.clear();

            all_labels.clear();
            // first layer
            auto first_layer = *data_batch * *first_weight;
            auto frst = first_layer->broadcast_add(*first_bias);
            auto first_layer_sigmoid = sigmoid(*first_layer);

            // second layer
            auto second_layer = *first_layer_sigmoid * *second_weight;
           auto sec = second_layer->broadcast_add(*second_bias);

            ; // Stop after reading n lines
        }
        ++lines_read;
        if (lines_read == 31) {
            break;
        }


        // Optional: limit for testing
        // if (sample_count == 1000) break;
    }
    for (int i = 0; i < all_data.size(); ++i) {
        std::cout << all_data[i] << " ";
    }
    std::cout << all_data.size() << "\n";
    std::cout << all_labels[0] << "\n";
    std::cout << "\n";
    file.close();


    return 0;
}