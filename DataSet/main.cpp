#include <iostream>
#include <memory>
#include <vector>
#include <chrono> // Added for timing
#include "./dataset.h"
#include "../AutoGrad/AutoGradTensor.h"
void print_data(const AutogradTensor<float>& tensor) {
    std::cout << "Data: ";
    for (size_t i = 0; i < tensor.data().size(); ++i) {
        std::cout << tensor.data()[i] << " ";
    }
    std::cout << "\n";
    
}
void print_shape(const AutogradTensor<float>& tensor) {
    std::cout << "Shape: [";
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
        std::cout << tensor.shape()[i];
        if (i < tensor.shape().size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}
int main() {
    try {
        auto dataset = std::make_shared<MNISTDataset<float>>("mnist_train.csv");
        std::cout << "Dataset loaded with " << dataset->size() << " samples\n";
        
        size_t batch_size = 32;
        bool shuffle = false;
        size_t num_workers = 5;
        
        DataLoader<float> dataloader(dataset, batch_size, shuffle, num_workers);

        // std::pair<AutogradTensor<float>, AutogradTensor<float>> batch;
        // bool sucess=dataloader.next(batch);
        // print_data(batch.second);
        // print_shape(batch.first);
        // std::cout << "True or nat: "  << sucess<< "\n";


        
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
                   // Test individual sample access
                std::cout << "\n--- Individual Sample Access ---\n";
                auto sample_image = dataset->get_item(0);
                auto sample_label = dataset->get_label(0);
                
                std::cout << "Sample image shape: [";
                for (size_t i = 0; i < sample_image.shape().size(); ++i) {
                    std::cout << sample_image.shape()[i];
                    if (i < sample_image.shape().size() - 1) std::cout << ", ";
                }
                std::cout << "]\n";
                std::cout << "Sample label: " << static_cast<int>(sample_label.data()[0]) << "\n";
                
                std::cout << "First 10 pixel values: ";
                const auto& pixel_data = sample_image.data();
                for (size_t i = 0; i < std::min(10UL, pixel_data.size()); ++i) {
                    std::cout << pixel_data[i] << " ";
                }
                std::cout << "\n";
                batch_count++;
                if(batch_count>3){
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