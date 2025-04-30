#include "encrypted_transformer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

EncryptedTransformerWeights::EncryptedTransformerWeights(std::shared_ptr<seal::SEALContext> context)
    : context_(context) {
    // Initialize weights containers
}

void EncryptedTransformerWeights::loadFromPretrained(
    const std::string& model_path,
    seal::Encryptor& encryptor,
    seal::CKKSEncoder& encoder,
    double scale) {
    
    std::cout << "Loading pretrained weights from: " << model_path << std::endl;
    
    try {
        // Load the weight matrices from files
        std::vector<std::vector<double>> wq_data = loadWeightsFromFile(model_path + "/wq.bin");
        std::vector<std::vector<double>> wk_data = loadWeightsFromFile(model_path + "/wk.bin");
        std::vector<std::vector<double>> wv_data = loadWeightsFromFile(model_path + "/wv.bin");
        std::vector<std::vector<double>> wo_data = loadWeightsFromFile(model_path + "/wo.bin");
        std::vector<std::vector<double>> ff1_data = loadWeightsFromFile(model_path + "/ff1.bin");
        std::vector<std::vector<double>> ff2_data = loadWeightsFromFile(model_path + "/ff2.bin");
        
        std::cout << "Successfully loaded weight matrices:" << std::endl;
        std::cout << "- Query weights: " << wq_data.size() << "x" << (wq_data.empty() ? 0 : wq_data[0].size()) << std::endl;
        std::cout << "- Key weights: " << wk_data.size() << "x" << (wk_data.empty() ? 0 : wk_data[0].size()) << std::endl;
        std::cout << "- Value weights: " << wv_data.size() << "x" << (wv_data.empty() ? 0 : wv_data[0].size()) << std::endl;
        std::cout << "- Output weights: " << wo_data.size() << "x" << (wo_data.empty() ? 0 : wo_data[0].size()) << std::endl;
        std::cout << "- FF1 weights: " << ff1_data.size() << "x" << (ff1_data.empty() ? 0 : ff1_data[0].size()) << std::endl;
        std::cout << "- FF2 weights: " << ff2_data.size() << "x" << (ff2_data.empty() ? 0 : ff2_data[0].size()) << std::endl;
        
        // Encrypt the weights
        std::cout << "Encrypting weights..." << std::endl;
        
        // Helper function to encrypt a matrix
        auto encryptMatrix = [&](const std::vector<std::vector<double>>& matrix) -> seal::Ciphertext {
            std::vector<double> flattened;
            for (const auto& row : matrix) {
                flattened.insert(flattened.end(), row.begin(), row.end());
            }
            
            seal::Plaintext plain;
            encoder.encode(flattened, scale, plain);
            
            seal::Ciphertext encrypted;
            encryptor.encrypt(plain, encrypted);
            
            return encrypted;
        };
        
        // Encrypt query weights
        wq_weights.clear();
        for (size_t i = 0; i < wq_data.size(); i++) {
            std::vector<double> row = wq_data[i];
            seal::Plaintext plain;
            encoder.encode(row, scale, plain);
            
            seal::Ciphertext encrypted;
            encryptor.encrypt(plain, encrypted);
            wq_weights.push_back(encrypted);
        }
        
        // Encrypt key weights
        wk_weights.clear();
        for (size_t i = 0; i < wk_data.size(); i++) {
            std::vector<double> row = wk_data[i];
            seal::Plaintext plain;
            encoder.encode(row, scale, plain);
            
            seal::Ciphertext encrypted;
            encryptor.encrypt(plain, encrypted);
            wk_weights.push_back(encrypted);
        }
        
        // Encrypt value weights
        wv_weights.clear();
        for (size_t i = 0; i < wv_data.size(); i++) {
            std::vector<double> row = wv_data[i];
            seal::Plaintext plain;
            encoder.encode(row, scale, plain);
            
            seal::Ciphertext encrypted;
            encryptor.encrypt(plain, encrypted);
            wv_weights.push_back(encrypted);
        }
        
        // Encrypt output weights
        wo_weights.clear();
        for (size_t i = 0; i < wo_data.size(); i++) {
            std::vector<double> row = wo_data[i];
            seal::Plaintext plain;
            encoder.encode(row, scale, plain);
            
            seal::Ciphertext encrypted;
            encryptor.encrypt(plain, encrypted);
            wo_weights.push_back(encrypted);
        }
        
        // Encrypt FF1 weights
        ff1_weights.clear();
        for (size_t i = 0; i < ff1_data.size(); i++) {
            std::vector<double> row = ff1_data[i];
            seal::Plaintext plain;
            encoder.encode(row, scale, plain);
            
            seal::Ciphertext encrypted;
            encryptor.encrypt(plain, encrypted);
            ff1_weights.push_back(encrypted);
        }
        
        // Encrypt FF2 weights
        ff2_weights.clear();
        for (size_t i = 0; i < ff2_data.size(); i++) {
            std::vector<double> row = ff2_data[i];
            seal::Plaintext plain;
            encoder.encode(row, scale, plain);
            
            seal::Ciphertext encrypted;
            encryptor.encrypt(plain, encrypted);
            ff2_weights.push_back(encrypted);
        }
        
        std::cout << "Successfully encrypted all weight matrices" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        
        // Generate random weights if loading fails
        std::cout << "Falling back to random weights for testing" << std::endl;
        
        // Random weight generation helper
        auto generateRandomWeights = [&](int rows, int cols) -> std::vector<seal::Ciphertext> {
            std::vector<seal::Ciphertext> weights;
            std::srand(static_cast<unsigned int>(std::time(nullptr)));
            
            for (int i = 0; i < rows; i++) {
                std::vector<double> row(cols);
                for (int j = 0; j < cols; j++) {
                    // Generate values between -0.1 and 0.1
                    row[j] = (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 0.2;
                }
                
                seal::Plaintext plain;
                encoder.encode(row, scale, plain);
                
                seal::Ciphertext encrypted;
                encryptor.encrypt(plain, encrypted);
                weights.push_back(encrypted);
            }
            
            return weights;
        };
        
        // Generate random weights for each matrix
        int hidden_size = 128; // Default for testing
        int intermediate_size = 512;
        
        wq_weights = generateRandomWeights(hidden_size, hidden_size);
        wk_weights = generateRandomWeights(hidden_size, hidden_size);
        wv_weights = generateRandomWeights(hidden_size, hidden_size);
        wo_weights = generateRandomWeights(hidden_size, hidden_size);
        ff1_weights = generateRandomWeights(hidden_size, intermediate_size);
        ff2_weights = generateRandomWeights(intermediate_size, hidden_size);
        
        std::cout << "Generated random weight matrices for testing" << std::endl;
    }
}

std::vector<std::vector<double>> EncryptedTransformerWeights::loadWeightsFromFile(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open weight file: " + file_path);
    }
    
    // Read dimensions
    int rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    
    // Allocate matrix
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    
    // Read data
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double value;
            file.read(reinterpret_cast<char*>(&value), sizeof(double));
            matrix[i][j] = value;
        }
    }
    
    file.close();
    return matrix;
} 