#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <memory>
#include "../AutoGrad/AutoGradTensor.h"

// Abstract Dataset class
template <typename T>
class Dataset {
public:
    virtual AutogradTensor<T> get_item(size_t index) = 0;
    virtual AutogradTensor<T> get_label(size_t index) = 0;
    virtual size_t size() const = 0;
    virtual ~Dataset() = default;
};

// MNISTDataset implementation
template <typename T>
class MNISTDataset : public Dataset<T> {
private:
    std::vector<std::vector<T>> images_;
    std::vector<T> labels_;
    size_t num_images_;
    size_t rows_ = 28;
    size_t cols_ = 28;

public:
    MNISTDataset(const std::string& csv_file);

    AutogradTensor<T> get_item(size_t index) override;
    AutogradTensor<T> get_label(size_t index) override;
    size_t size() const override;
};

// DataLoader class
template <typename T>
class DataLoader {
private:
    std::shared_ptr<Dataset<T>> dataset_;
    size_t batch_size_;
    bool shuffle_;
    size_t num_workers_;
    std::vector<size_t> indices_;
    std::atomic<size_t> current_idx_;
    std::queue<std::pair<AutogradTensor<T>, AutogradTensor<T>>> batch_queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
    std::atomic<bool> stop_workers_;
    std::vector<std::thread> workers_;

    void worker_thread();

public:
    DataLoader(std::shared_ptr<Dataset<T>> dataset, size_t batch_size, bool shuffle = false, size_t num_workers = 0);
    ~DataLoader();

    bool next(std::pair<AutogradTensor<T>, AutogradTensor<T>>& batch);
    void reset();
};

// Include implementation
#include "dataset.tpp"
