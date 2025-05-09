#include "encrypted_transformer.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>

EncryptedTransformerWeights::EncryptedTransformerWeights(std::shared_ptr<seal::SEALContext> context)
    : context_(context) {
}

void EncryptedTransformerWeights::loadFromPretrained(
    const std::string& model_path,
    seal::Encryptor& encryptor,
    seal::CKKSEncoder& encoder,
    double scale) {
    
    try {
        // Load query weight
        std::string wq_path = model_path + "/wq.bin";
        std::ifstream wq_file(wq_path, std::ios::binary);
        if (!wq_file.is_open()) {
            throw std::runtime_error("Failed to open query weight file: " + wq_path);
        }
        
        // Read dimensions
        int rows, cols;
        wq_file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        wq_file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        
        // Read weight values
        std::vector<double> wq_values(rows * cols);
        for (int i = 0; i < rows * cols; i++) {
            double val;
            wq_file.read(reinterpret_cast<char*>(&val), sizeof(double));
            wq_values[i] = val;
        }
        
        // Encode as plaintext
        encoder.encode(wq_values, scale, query_weight_);
        
        // Similarly load other weights
        std::string wk_path = model_path + "/wk.bin";
        std::ifstream wk_file(wk_path, std::ios::binary);
        if (!wk_file.is_open()) {
            throw std::runtime_error("Failed to open key weight file: " + wk_path);
        }
        
        wk_file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        wk_file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        
        std::vector<double> wk_values(rows * cols);
        for (int i = 0; i < rows * cols; i++) {
            double val;
            wk_file.read(reinterpret_cast<char*>(&val), sizeof(double));
            wk_values[i] = val;
        }
        
        encoder.encode(wk_values, scale, key_weight_);
        
        // Load value weight
        std::string wv_path = model_path + "/wv.bin";
        std::ifstream wv_file(wv_path, std::ios::binary);
        if (!wv_file.is_open()) {
            throw std::runtime_error("Failed to open value weight file: " + wv_path);
        }
        
        wv_file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        wv_file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        
        std::vector<double> wv_values(rows * cols);
        for (int i = 0; i < rows * cols; i++) {
            double val;
            wv_file.read(reinterpret_cast<char*>(&val), sizeof(double));
            wv_values[i] = val;
        }
        
        encoder.encode(wv_values, scale, value_weight_);
        
        // Load output weight
        std::string wo_path = model_path + "/wo.bin";
        std::ifstream wo_file(wo_path, std::ios::binary);
        if (!wo_file.is_open()) {
            throw std::runtime_error("Failed to open output weight file: " + wo_path);
        }
        
        wo_file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        wo_file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        
        std::vector<double> wo_values(rows * cols);
        for (int i = 0; i < rows * cols; i++) {
            double val;
            wo_file.read(reinterpret_cast<char*>(&val), sizeof(double));
            wo_values[i] = val;
        }
        
        encoder.encode(wo_values, scale, output_weight_);
        
        // Load feed-forward weights
        std::string ff1_path = model_path + "/ff1.bin";
        std::ifstream ff1_file(ff1_path, std::ios::binary);
        if (!ff1_file.is_open()) {
            throw std::runtime_error("Failed to open FF1 weight file: " + ff1_path);
        }
        
        ff1_file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        ff1_file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        
        std::vector<double> ff1_values(rows * cols);
        for (int i = 0; i < rows * cols; i++) {
            double val;
            ff1_file.read(reinterpret_cast<char*>(&val), sizeof(double));
            ff1_values[i] = val;
        }
        
        encoder.encode(ff1_values, scale, ff1_weight_);
        
        std::string ff2_path = model_path + "/ff2.bin";
        std::ifstream ff2_file(ff2_path, std::ios::binary);
        if (!ff2_file.is_open()) {
            throw std::runtime_error("Failed to open FF2 weight file: " + ff2_path);
        }
        
        ff2_file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        ff2_file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        
        std::vector<double> ff2_values(rows * cols);
        for (int i = 0; i < rows * cols; i++) {
            double val;
            ff2_file.read(reinterpret_cast<char*>(&val), sizeof(double));
            ff2_values[i] = val;
        }
        
        encoder.encode(ff2_values, scale, ff2_weight_);
        
        std::cout << "Successfully loaded all weight matrices" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "Error loading pretrained weights: " << e.what() << std::endl;
        std::cout << "Generating minimal random weights instead" << std::endl;
        
        // Create minimal dummy weights when loading fails
        createDummyWeights(64, encryptor, encoder, scale);
    }
}

// Load weights from a binary file
std::vector<std::vector<double>> EncryptedTransformerWeights::loadWeightsFromFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open weight file: " + filepath);
    }
    
    int rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double val;
            file.read(reinterpret_cast<char*>(&val), sizeof(double));
            matrix[i][j] = val;
        }
    }
    
    file.close();
    return matrix;
}

// Create dummy weights for testing
void EncryptedTransformerWeights::createDummyWeights(
    int hidden_size,
    seal::Encryptor& encryptor,
    seal::CKKSEncoder& encoder,
    double scale) {
    
    std::cout << "Creating minimal dummy weights for hidden_size = " << hidden_size << std::endl;
    
    // Use a fixed seed for reproducibility
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-0.01, 0.01);
    
    // Create small-valued weight matrices
    int intermediate_size = hidden_size * 4;
    
    // Query weights: hidden_size x hidden_size
    std::vector<double> wq_values(hidden_size * hidden_size, 0.0);
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            // Initialize with small values
            wq_values[i * hidden_size + j] = (i == j) ? 0.1 : dist(gen);
        }
    }
    encoder.encode(wq_values, scale, query_weight_);
    
    // Key weights: hidden_size x hidden_size
    std::vector<double> wk_values(hidden_size * hidden_size, 0.0);
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            wk_values[i * hidden_size + j] = (i == j) ? 0.1 : dist(gen);
        }
    }
    encoder.encode(wk_values, scale, key_weight_);
    
    // Value weights: hidden_size x hidden_size
    std::vector<double> wv_values(hidden_size * hidden_size, 0.0);
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            wv_values[i * hidden_size + j] = (i == j) ? 0.1 : dist(gen);
        }
    }
    encoder.encode(wv_values, scale, value_weight_);
    
    // Output weights: hidden_size x hidden_size
    std::vector<double> wo_values(hidden_size * hidden_size, 0.0);
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            wo_values[i * hidden_size + j] = (i == j) ? 0.1 : dist(gen);
        }
    }
    encoder.encode(wo_values, scale, output_weight_);
    
    // FF1 weights: hidden_size x intermediate_size
    std::vector<double> ff1_values(hidden_size * intermediate_size, 0.0);
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < intermediate_size; j++) {
            ff1_values[i * intermediate_size + j] = (i == j % hidden_size) ? 0.1 : dist(gen);
        }
    }
    encoder.encode(ff1_values, scale, ff1_weight_);
    
    // FF2 weights: intermediate_size x hidden_size
    std::vector<double> ff2_values(intermediate_size * hidden_size, 0.0);
    for (int i = 0; i < intermediate_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            ff2_values[i * hidden_size + j] = (i % hidden_size == j) ? 0.1 : dist(gen);
        }
    }
    encoder.encode(ff2_values, scale, ff2_weight_);
    
    std::cout << "Dummy weights created successfully" << std::endl;
} 