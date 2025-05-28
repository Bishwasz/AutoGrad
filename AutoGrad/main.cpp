#include "AutoGradTensor.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>

// Helper function to load MNIST data from CSV (images and labels)
void load_mnist_data(const std::string& image_file, const std::string& label_file,
                     std::vector<std::vector<float>>& images, std::vector<std::vector<float>>& labels) {
    std::ifstream img_stream(image_file);
    std::ifstream lbl_stream(label_file);
    if (!img_stream.is_open() || !lbl_stream.is_open()) {
        throw std::runtime_error("Failed to open MNIST files");
    }

    std::string line;
    std::getline(img_stream, line);
    std::getline(lbl_stream, line);

    images.clear();
    labels.clear();

    while (std::getline(img_stream, line)) {
        std::vector<float> image(784);
        std::string pixel;
        std::stringstream img_ss(line);
        if (!std::getline(img_ss, pixel, ',')) {
            throw std::runtime_error("Failed to read label from image line");
        }

        int idx = 0;
        while (std::getline(img_ss, pixel, ',') && idx < 784) {
            try {
                image[idx++] = std::stof(pixel) / 255.0f;
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid pixel value at index " + std::to_string(idx));
            }
        }
        if (idx != 784) {
            throw std::runtime_error("Incomplete image data: expected 784 pixels, got " + std::to_string(idx));
        }

        if (!std::getline(lbl_stream, line)) {
            throw std::runtime_error("Mismatch in number of labels and images");
        }
        std::vector<float> label(10, 0.0f);
        try {
            int label_idx = std::stoi(line);
            if (label_idx < 0 || label_idx > 9) {
                throw std::runtime_error("Invalid label value: " + line);
            }
            label[label_idx] = 1.0f;
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid label format: " + line);
        }

        images.push_back(image);
        labels.push_back(label);
    }

    if (!lbl_stream.eof()) {
        throw std::runtime_error("More labels than images");
    }
}

// Neural network model
class MNISTModel {
public:
    AutogradTensor<float> W1, b1, W2, b2, W3, b3;

    MNISTModel() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.01f);

        std::vector<float> w1_data(784 * 256);
        std::vector<float> b1_data(256);
        for (auto& w : w1_data) w = dist(gen);
        for (auto& b : b1_data) b = dist(gen);
        W1 = AutogradTensor<float>({784, 256}, w1_data, true);
        b1 = AutogradTensor<float>({256}, b1_data, true);

        std::vector<float> w2_data(256 * 128);
        std::vector<float> b2_data(128);
        for (auto& w : w2_data) w = dist(gen);
        for (auto& b : b2_data) b = dist(gen);
        W2 = AutogradTensor<float>({256, 128}, w2_data, true);
        b2 = AutogradTensor<float>({128}, b2_data, true);

        std::vector<float> w3_data(128 * 10);
        std::vector<float> b3_data(10);
        for (auto& w : w3_data) w = dist(gen);
        for (auto& b : b3_data) b = dist(gen);
        W3 = AutogradTensor<float>({128, 10}, w3_data, true);
        b3 = AutogradTensor<float>({10}, b3_data, true);
    }

    AutogradTensor<float> forward(AutogradTensor<float>& input) {
        auto h1 = input * W1;
        h1 = h1.broadcast_add(b1);
        h1 = h1.relu();

        auto h2 = h1 * W2;
        h2 = h2.broadcast_add(b2);
        h2 = h2.relu();

        auto out = h2 * W3;
        out = out.broadcast_add(b3);
        return out;
    }

    void update_parameters(float lr) {
        for (size_t i = 0; i < W1.grad().size(); ++i) {
            (W1.data())[i] -= lr * W1.grad()[i];
        }
        for (size_t i = 0; i < b1.grad().size(); ++i) {
            (b1.data())[i] -= lr * b1.grad()[i];
        }
        for (size_t i = 0; i < W2.grad().size(); ++i) {
            (W2.data())[i] -= lr * W2.grad()[i];
        }
        for (size_t i = 0; i < b2.grad().size(); ++i) {
            (b2.data())[i] -= lr * b2.grad()[i];
        }
        for (size_t i = 0; i < W3.grad().size(); ++i) {
            (W3.data())[i] -= lr * W3.grad()[i];
        }
        for (size_t i = 0; i < b3.grad().size(); ++i) {
            (b3.data())[i] -= lr * b3.grad()[i];
        }
    }
};

int main() {
    std::vector<std::vector<float>> train_images, train_labels;
    std::vector<std::vector<float>> test_images, test_labels;
    try {
        load_mnist_data("mnist_train.csv", "mnist_train_labels.csv", train_images, train_labels);
        load_mnist_data("mnist_test.csv", "mnist_test_labels.csv", test_images, test_labels);
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return 1;
    }

//     const int batch_size = 32;
//     const int epochs = 1;
//     const float learning_rate = 0.01f;
//     const int num_batches = train_images.size() / batch_size;

//     MNISTModel model;
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dist(0, train_images.size() - 1);

//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         float total_loss = 0.0f;
//         int correct = 0;

//         for (int batch = 0; batch < num_batches; ++batch) {
//             std::vector<float> batch_images(batch_size * 784);
//             std::vector<float> batch_labels(batch_size * 10);
//             for (int i = 0; i < batch_size; ++i) {
//                 int idx = dist(gen);
//                 for (int j = 0; j < 784; ++j) {
//                     batch_images[i * 784 + j] = train_images[idx][j];
//                 }
//                 for (int j = 0; j < 10; ++j) {
//                     batch_labels[i * 10 + j] = train_labels[idx][j];
//                 }
//             }

//             AutogradTensor<float> input({batch_size, 784}, batch_images, true);
//             AutogradTensor<float> labels({batch_size, 10}, batch_labels, false);

//             // Forward pass
//             auto output = model.forward(input);
            
//             // Use result_ptr to store output (example usage)
//             std::unique_ptr<AutogradTensor<float>> result_ptr = std::make_unique<AutogradTensor<float>>(output.shape(), output.data(), output.requires_grad());
//             *result_ptr = output;

//             auto loss = result_ptr->cross_entropy_loss(labels);

//             model.W1.zero_grad();
//             model.b1.zero_grad();
//             model.W2.zero_grad();
//             model.b2.zero_grad();
//             model.W3.zero_grad ();
//             model.b3.zero_grad();
//             loss.backward();

//             model.update_parameters(learning_rate);

//             total_loss += (*loss.data())[0];
//             for (int i = 0; i < batch_size; ++i) {
//                 int pred = 0;
//                 float max_val = (*output.data())[i * 10];
//                 for (int j = 1; j < 10; ++j) {
//                     if ((*output.data())[i * 10 + j] > max_val) {
//                         max_val = (*output.data())[i * 10 + j];
//                         pred = j;
//                     }
//                 }
//                 if (batch_labels[i * 10 + pred] == 1.0f) {
//                     correct++;
//                 }
//             }
//         }

//         std::cout << "Epoch " << epoch + 1 << "/" << epochs
//                   << ", Loss: " << total_loss / num_batches
//                   << ", Accuracy: " << (float)correct / (num_batches * batch_size) * 100 << "%\n";
//     }

//     float test_loss = 0.0f;
//     int correct = 0;
//     for (size_t i = 0; i < test_images.size(); i += batch_size) {
//         int current_batch_size = std::min(batch_size, (int)(test_images.size() - i));
//         std::vector<float> batch_images(current_batch_size * 784);
//         std::vector<float> batch_labels(current_batch_size * 10);
//         for (int j = 0; j < current_batch_size; ++j) {
//             for (int k = 0; k < 784; ++k) {
//                 batch_images[j * 784 + k] = test_images[i + j][k];
//             }
//             for (int k = 0; k < 10; ++k) {
//                 batch_labels[j * 10 + k] = test_labels[i + j][k];
//             }
//         }

//         AutogradTensor<float> input({current_batch_size, 784}, batch_images, false);
//         AutogradTensor<float> labels({current_batch_size, 10}, batch_labels, false);
//         auto output = model.forward(input);
//         auto loss = output.cross_entropy_loss(labels);

//         test_loss += (*loss.data())[0] * current_batch_size;
//         for (int j = 0; j < current_batch_size; ++j) {
//             int pred = 0;
//             float max_val = (*output

// .data())[j * 10];
//             for (int k = 1; k < 10; ++k) {
//                 if ((*output.data())[j * 10 + k] > max_val) {
//                     max_val = (*output.data())[j * 10 + k];
//                     pred = k;
//                 }
//             }
//             if (batch_labels[j * 10 + pred] == 1.0f) {
//                 correct++;
//             }
//         }
//     }

//     std::cout << "Test Loss: " << test_loss / test_images.size()
//               << ", Test Accuracy: " << (float)correct / test_images.size() * 100 << "%\n";

//     return 0;
}