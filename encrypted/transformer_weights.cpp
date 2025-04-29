#include "encrypted_transformer.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>

EncryptedTransformerWeights::EncryptedTransformerWeights(heongpu::HEContext<heongpu::Scheme::CKKS>& context)
    : context_(context) {
}

void EncryptedTransformerWeights::loadFromPretrained(
    const std::string& model_path,
    heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor,
    heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
    double scale) {
    
    std::cout << "Loading pretrained weights from: " << model_path << std::endl;
    
    try {
        // Load weights for each layer
        // Query weights
        std::vector<std::vector<double>> raw_wq = loadWeightsFromFile(model_path + "/wq.bin");
        std::cout << "Loaded query weights: " << raw_wq.size() << " layers" << std::endl;
        
        // Key weights
        std::vector<std::vector<double>> raw_wk = loadWeightsFromFile(model_path + "/wk.bin");
        std::cout << "Loaded key weights: " << raw_wk.size() << " layers" << std::endl;
        
        // Value weights
        std::vector<std::vector<double>> raw_wv = loadWeightsFromFile(model_path + "/wv.bin");
        std::cout << "Loaded value weights: " << raw_wv.size() << " layers" << std::endl;
        
        // Output projection weights
        std::vector<std::vector<double>> raw_wo = loadWeightsFromFile(model_path + "/wo.bin");
        std::cout << "Loaded output weights: " << raw_wo.size() << " layers" << std::endl;
        
        // Feed-forward weights
        std::vector<std::vector<double>> raw_ff1 = loadWeightsFromFile(model_path + "/ff1.bin");
        std::cout << "Loaded FF1 weights: " << raw_ff1.size() << " layers" << std::endl;
        
        std::vector<std::vector<double>> raw_ff2 = loadWeightsFromFile(model_path + "/ff2.bin");
        std::cout << "Loaded FF2 weights: " << raw_ff2.size() << " layers" << std::endl;
        
        // Set execution options for GPU operations
        heongpu::ExecutionOptions options;
        options.set_storage_type(heongpu::storage_type::DEVICE)
               .set_initial_location(true);
        
        // Encrypt weights for each layer
        std::cout << "Encrypting weights..." << std::endl;
        
        // Clear previous weights if any
        wq_weights.clear();
        wk_weights.clear();
        wv_weights.clear();
        wo_weights.clear();
        ff1_weights.clear();
        ff2_weights.clear();
        
        // For each layer
        for (size_t layer = 0; layer < raw_wq.size(); ++layer) {
            std::cout << "Encrypting layer " << layer + 1 << "/" << raw_wq.size() << std::endl;
            
            // Query weights
            heongpu::Plaintext<heongpu::Scheme::CKKS> wq_plain(context_);
            encoder.encode(wq_plain, raw_wq[layer], scale);
            
            heongpu::Ciphertext<heongpu::Scheme::CKKS> wq_cipher(context_);
            encryptor.encrypt(wq_cipher, wq_plain);
            
            wq_weights.push_back(wq_cipher);
            
            // Key weights
            heongpu::Plaintext<heongpu::Scheme::CKKS> wk_plain(context_);
            encoder.encode(wk_plain, raw_wk[layer], scale);
            
            heongpu::Ciphertext<heongpu::Scheme::CKKS> wk_cipher(context_);
            encryptor.encrypt(wk_cipher, wk_plain);
            
            wk_weights.push_back(wk_cipher);
            
            // Value weights
            heongpu::Plaintext<heongpu::Scheme::CKKS> wv_plain(context_);
            encoder.encode(wv_plain, raw_wv[layer], scale);
            
            heongpu::Ciphertext<heongpu::Scheme::CKKS> wv_cipher(context_);
            encryptor.encrypt(wv_cipher, wv_plain);
            
            wv_weights.push_back(wv_cipher);
            
            // Output weights
            heongpu::Plaintext<heongpu::Scheme::CKKS> wo_plain(context_);
            encoder.encode(wo_plain, raw_wo[layer], scale);
            
            heongpu::Ciphertext<heongpu::Scheme::CKKS> wo_cipher(context_);
            encryptor.encrypt(wo_cipher, wo_plain);
            
            wo_weights.push_back(wo_cipher);
            
            // FF1 weights
            heongpu::Plaintext<heongpu::Scheme::CKKS> ff1_plain(context_);
            encoder.encode(ff1_plain, raw_ff1[layer], scale);
            
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ff1_cipher(context_);
            encryptor.encrypt(ff1_cipher, ff1_plain);
            
            ff1_weights.push_back(ff1_cipher);
            
            // FF2 weights
            heongpu::Plaintext<heongpu::Scheme::CKKS> ff2_plain(context_);
            encoder.encode(ff2_plain, raw_ff2[layer], scale);
            
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ff2_cipher(context_);
            encryptor.encrypt(ff2_cipher, ff2_plain);
            
            ff2_weights.push_back(ff2_cipher);
        }
        
        std::cout << "Successfully encrypted all weights." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading pretrained weights: " << e.what() << std::endl;
        throw;
    }
}

std::vector<std::vector<double>> EncryptedTransformerWeights::loadWeightsFromFile(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open weight file: " + file_path);
    }
    
    // In a real-world scenario, this would read binary weight files in a specific format
    // For this example implementation, we'll use a simple placeholder format:
    // First 4 bytes: number of layers (uint32_t)
    // For each layer:
    //   - 4 bytes: number of elements (uint32_t)
    //   - 8 * num_elements bytes: elements (double)
    
    uint32_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    
    std::vector<std::vector<double>> weights;
    weights.reserve(num_layers);
    
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        uint32_t num_elements;
        file.read(reinterpret_cast<char*>(&num_elements), sizeof(num_elements));
        
        std::vector<double> layer_weights(num_elements);
        file.read(reinterpret_cast<char*>(layer_weights.data()), 
                 num_elements * sizeof(double));
        
        weights.push_back(std::move(layer_weights));
    }
    
    return weights;
} 