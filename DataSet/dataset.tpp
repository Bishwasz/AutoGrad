#pragma once

#include "dataset.h"

// ==== MNISTDataset Implementation ====

template <typename T>
MNISTDataset<T>::MNISTDataset(const std::string& csv_file) {
    static_assert(std::is_floating_point<T>::value, "MNISTDataset requires floating-point type");
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + csv_file);
    }

    std::string line;
    std::getline(file, line); // Skip header
    size_t line_number = 2;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        if (!std::getline(ss, value, ',')) {
            throw std::runtime_error("Missing label in CSV at line " + std::to_string(line_number));
        }

        try {
            int label = std::stoi(value);
            if (label < 0 || label > 9)
                throw std::runtime_error("Label out of range at line " + std::to_string(line_number));
            labels_.push_back(static_cast<T>(label));
        } catch (...) {
            throw std::runtime_error("Invalid label in CSV at line " + std::to_string(line_number));
        }

        std::vector<T> pixels;
        pixels.reserve(rows_ * cols_);
        for (size_t i = 0; i < rows_ * cols_; ++i) {
            if (!std::getline(ss, value, ',')) {
                throw std::runtime_error("Insufficient pixels in CSV at line " + std::to_string(line_number));
            }
            try {
                pixels.push_back(static_cast<T>(std::stof(value)) / static_cast<T>(255.0));
            } catch (...) {
                throw std::runtime_error("Invalid pixel value at line " + std::to_string(line_number));
            }
        }

        images_.push_back(pixels);
        line_number++;
    }

    file.close();

    num_images_ = images_.size();
    if (num_images_ == 0 || images_[0].size() != rows_ * cols_) {
        throw std::runtime_error("Invalid dataset shape");
    }
}

template <typename T>
AutogradTensor<T> MNISTDataset<T>::get_item(size_t index) {
    if (index >= num_images_) throw std::out_of_range("Index out of range");
    return AutogradTensor<T>({ static_cast<int>(rows_)*static_cast<int>(cols_)}, images_[index], true);
}

template <typename T>
AutogradTensor<T> MNISTDataset<T>::get_label(size_t index) {
    if (index >= num_images_) throw std::out_of_range("Index out of range");
    return AutogradTensor<T>({1}, {labels_[index]}, true);
}

template <typename T>
size_t MNISTDataset<T>::size() const {
    return num_images_;
}

// ==== DataLoader Implementation ====

template <typename T>
void DataLoader<T>::worker_thread() {
    while (true) {
        size_t start_idx;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_workers_.load() || current_idx_.load() >= indices_.size()) return;
            start_idx = current_idx_.load();
            current_idx_ += batch_size_;
        }

        size_t actual_batch_size = std::min(batch_size_, indices_.size() - start_idx);
        if (actual_batch_size == 0) return;

        std::vector<AutogradTensor<T>> batch_data, batch_labels;
        batch_data.reserve(actual_batch_size);
        batch_labels.reserve(actual_batch_size);

        for (size_t i = 0; i < actual_batch_size; ++i) {
            batch_data.push_back(dataset_->get_item(indices_[start_idx + i]));
            batch_labels.push_back(dataset_->get_label(indices_[start_idx + i]));
        }


        std::vector<T> batched_data;
        for (const auto& tensor : batch_data) {
            batched_data.insert(batched_data.end(), tensor.data().begin(), tensor.data().end());
        }
        int column=batch_data[0].shape()[0];
        AutogradTensor<T> data_tensor({static_cast<int>(batch_size_), static_cast<int>(column)}, batched_data, true);

        std::vector<int> label_shape = batch_labels[0].shape();
        label_shape[0] = static_cast<int>(actual_batch_size);

        std::vector<T> batched_labels;
        for (const auto& tensor : batch_labels) {
            batched_labels.insert(batched_labels.end(), tensor.data().begin(), tensor.data().end());
        }

        AutogradTensor<T> label_tensor(label_shape, batched_labels, true);

        {
            std::unique_lock<std::mutex> lock(mutex_);
            batch_queue_.emplace(std::move(data_tensor), std::move(label_tensor));
            cond_var_.notify_one();
        }
    }
}

template <typename T>
DataLoader<T>::DataLoader(std::shared_ptr<Dataset<T>> dataset, size_t batch_size, bool shuffle, size_t num_workers)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle), num_workers_(num_workers),
      current_idx_(0), stop_workers_(false) {
    if (!dataset || dataset->size() == 0) throw std::invalid_argument("Dataset is null or empty");
    if (batch_size == 0) throw std::invalid_argument("Batch size must be > 0");

    indices_.resize(dataset_->size());
    std::iota(indices_.begin(), indices_.end(), 0);
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), std::mt19937(std::random_device()()));
    }

    for (size_t i = 0; i < num_workers_; ++i) {
        workers_.emplace_back(&DataLoader::worker_thread, this);
    }
}

template <typename T>
DataLoader<T>::~DataLoader() {
    stop_workers_ = true;
    cond_var_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

template <typename T>
bool DataLoader<T>::next(std::pair<AutogradTensor<T>, AutogradTensor<T>>& batch) {
    if (num_workers_ == 0) {
        size_t start_idx = current_idx_.load();
        if (start_idx >= indices_.size()) return false;
        size_t actual_batch_size = std::min(batch_size_, indices_.size() - start_idx);
        std::vector<AutogradTensor<T>> batch_data, batch_labels;
        for (size_t i = 0; i < actual_batch_size; ++i) {
            batch_data.push_back(dataset_->get_item(indices_[start_idx + i]));
            batch_labels.push_back(dataset_->get_label(indices_[start_idx + i]));
        }

        std::vector<int> data_shape = batch_data[0].shape();
        data_shape[0] = static_cast<int>(actual_batch_size);
        std::vector<T> batched_data;
        for (const auto& t : batch_data) batched_data.insert(batched_data.end(), t.data().begin(), t.data().end());
        AutogradTensor<T> data_tensor(data_shape, batched_data, true);

        std::vector<int> label_shape = batch_labels[0].shape();
        label_shape[0] = static_cast<int>(actual_batch_size);
        std::vector<T> batched_labels;
        for (const auto& t : batch_labels) batched_labels.insert(batched_labels.end(), t.data().begin(), t.data().end());
        AutogradTensor<T> label_tensor(label_shape, batched_labels, true);

        batch = std::make_pair(std::move(data_tensor), std::move(label_tensor));
        current_idx_ += actual_batch_size;
        return true;
    } else {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this] {
            return !batch_queue_.empty() || (current_idx_.load() >= indices_.size() && batch_queue_.empty());
        });

        if (batch_queue_.empty()) return false;
        batch = std::move(batch_queue_.front());
        batch_queue_.pop();
        return true;
    }
}

template <typename T>
void DataLoader<T>::reset() {
    stop_workers_ = true;
    cond_var_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    workers_.clear();

    current_idx_ = 0;
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), std::mt19937(std::random_device()()));
    }

    stop_workers_ = false;
    for (size_t i = 0; i < num_workers_; ++i) {
        workers_.emplace_back(&DataLoader::worker_thread, this);
    }
}
