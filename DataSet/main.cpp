#include <iostream>
#include <memory>
#include <vector>
#include <chrono> // Added for timing
#include "./dataset.h"
#include "../AutoGrad/AutoGradTensor.h"
#include "../Loss/loss.hpp"
void print_data(const AutogradTensor<float>& tensor) {
    std::cout << "Data: ";
    if(tensor.shape().size()>1){
        for (size_t i = 0; i <  static_cast<size_t>(tensor.shape()[0]); ++i) {
            std::cout << "[";
            for (size_t j = 0; j < static_cast<size_t>(tensor.shape()[1]); ++j) {
                std::cout << tensor.at({static_cast<int>(i), static_cast<int>(j)});
                if (j < tensor.shape()[1] - 1) std::cout << ", ";
            }
            std::cout << "]";
            if (i < tensor.shape()[0] - 1) std::cout << "\n ";
        }
    } else {
        for (size_t i = 0; i < static_cast<size_t>(tensor.shape()[0]); ++i) {
            std::cout << tensor.at({static_cast<int>(i)});
            if (i < tensor.shape()[0] - 1) std::cout << ", ";

        }

    std::cout << "\n";
    
}}
void print_shape(const AutogradTensor<float>& tensor) {
    std::cout << "Shape: [";
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
        std::cout << tensor.shape()[i];
        if (i < tensor.shape().size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}
AutogradTensor<float> one_hot_encode(AutogradTensor<float>& label, int num_classes, int batchSize) {
    std::vector<float> data(num_classes*batchSize, 0.0f);
    for(size_t i = 0; i < batchSize; ++i) {
        int label_value = static_cast<int>(label.at({static_cast<int>(i)})); // Assuming label is a 2D tensor with shape [batchSize, 1]
        if (label_value < 0 || label_value >= num_classes) {
            throw std::out_of_range("Label value out of range for one-hot encoding");
        }
        data[i * num_classes + label_value] = 1.0f; // Set the corresponding class index to 1
    }
    return AutogradTensor<float>({batchSize, num_classes}, data, false);
}
int main() {
    try {
        auto dataset = std::make_shared<MNISTDataset<float>>("mnist_train.csv");
        std::cout << "Dataset loaded with " << dataset->size() << " samples\n";
        
        size_t batch_size = 32;
        bool shuffle = true;
        size_t num_workers = 4;
        
        DataLoader<float> dataloader(dataset, batch_size, shuffle,num_workers);

        AutogradTensor<float> weigh1({784,200},true);
        AutogradTensor<float> weigh2({200,10},true);
        AutogradTensor<float> bias1({200},true);
        AutogradTensor<float> bias2({10},true);

        int max_epochs = 1;
        for (int epoch = 0; epoch < max_epochs; ++epoch) {
            std::cout << "\n--- Epoch " << (epoch + 1) << " ---\n";
            
            // Start timing
            auto start_time = std::chrono::high_resolution_clock::now();
            
            int batch_count = 0;
        while (true) {
            std::pair<AutogradTensor<float>, AutogradTensor<float>> batch;
            if (!dataloader.next(batch)) {
                break;
            }
            // print_data(batch.first);
            // print_data(batch.second);

            auto layer1= batch.first * weigh1 ;
            auto layer1_int= layer1.broadcast_add(bias1);

            auto layer2=layer1_int.relu();
            auto prediction=layer2 * weigh2;
            auto prediction_int= prediction.broadcast_add(bias2);

            auto labels = one_hot_encode(batch.second, 10, batch.first.shape()[0]);

            auto loss = Loss<float>::cross_entropy_loss(prediction_int, labels);
            loss.backward();
            print_data(loss);

            weigh1 -= weigh1.grad() * 0.03f;
            weigh2 -= weigh2.grad() * 0.03f;
            bias1 -= bias1.grad() * 0.03f;
            bias2 -= bias2.grad() * 0.03f;

            weigh1.grad().zero_data(); // Reset gradients after each batch
            weigh2.grad().zero_data(); // Reset gradients after each batch
            bias1.grad().zero_data(); // Reset gradients after each batch
            bias2.grad().zero_data(); // Reset gradients after each batch

            batch_count++;
            if(batch_count>600) {
                break;
            }

        }
                
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Print timing results
            std::cout << "Processed " << batch_count << " batches in epoch " << (epoch + 1) 
                      << " in " << duration.count() << " milliseconds (" 
                      << duration.count() / 1000.0 << " seconds)\n";
            
            dataloader.reset();             
            }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}