#include <iostream>
#include <memory>
#include <vector>
#include "./dataset.h"
#include "../AutoGrad/AutoGradTensor.h"
int main() {
    try {
        auto dataset = std::make_shared<MNISTDataset<float>>("mnist_train.csv");
        std::cout << "Dataset loaded with " << dataset->size() << " samples\n";
        
        size_t batch_size = 32;
        bool shuffle = true;
        size_t num_workers = 2;
        
        DataLoader<float> dataloader(dataset, batch_size, shuffle, num_workers);
        
        int max_epochs = 3;
        for (int epoch = 0; epoch < max_epochs; ++epoch) {
            std::cout << "\n--- Epoch " << (epoch + 1) << " ---\n";
            
            int batch_count = 0;
            while (true) {
                std::pair<AutogradTensor<float>, AutogradTensor<float>> batch;
                if (!dataloader.next(batch)) {
                    break;
                }
                
                AutogradTensor<float>& data = batch.first;
                AutogradTensor<float>& labels = batch.second;
                
                std::cout << "Batch " << batch_count + 1 
                         << " - Data shape: [";
                for (size_t i = 0; i < data.shape().size(); ++i) {
                    std::cout << data.shape()[i];
                    if (i < data.shape().size() - 1) std::cout << ", ";
                }
                std::cout << "], Labels shape: [";
                for (size_t i = 0; i < labels.shape().size(); ++i) {
                    std::cout << labels.shape()[i];
                    if (i < labels.shape().size() - 1) std::cout << ", ";
                }
                std::cout << "]\n";
                
                std::cout << "First few labels: ";
                const auto& label_data = labels.data();
                for (size_t i = 0; i < std::min(5UL, label_data.size()); ++i) {
                    std::cout << static_cast<int>(label_data[i]) << " ";
                }
                std::cout << "\n";
                
                batch_count++;
                if (batch_count >= 5) break; // Demo limitation
            }
            
            std::cout << "Processed " << batch_count << " batches in epoch " << (epoch + 1) << "\n";
            dataloader.reset();
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
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}