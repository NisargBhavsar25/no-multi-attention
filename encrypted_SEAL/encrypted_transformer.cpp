#include "encrypted_transformer.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <random>

// Constructor for the weights
EncryptedTransformerWeights::EncryptedTransformerWeights(std::shared_ptr<seal::SEALContext> context)
    : context_(context) {
}

// Load weights from pre-trained model files
void EncryptedTransformerWeights::loadFromPretrained(const std::string& model_path,
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
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading pretrained weights: " + std::string(e.what()));
    }
}

// Create minimal dummy weights for testing
void EncryptedTransformerWeights::createDummyWeights(int hidden_size,
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

// Constructor for the transformer
EncryptedTransformer::EncryptedTransformer(
    std::shared_ptr<seal::SEALContext> context,
    const seal::RelinKeys& relin_keys,
    const seal::GaloisKeys& galois_keys,
    int num_layers,
    int hidden_size,
    int num_heads,
    double scale)
    : context_(context),
      relin_keys_(relin_keys),
      galois_keys_(galois_keys),
      num_layers_(num_layers),
      hidden_size_(hidden_size),
      num_heads_(num_heads),
      scale_(scale) {
}

void EncryptedTransformer::setWeights(std::shared_ptr<EncryptedTransformerWeights> weights) {
    weights_ = weights;
}

// Helper for noise management
void EncryptedTransformer::rescaleIfNeeded(
    seal::Ciphertext& cipher,
    seal::Evaluator& evaluator,
    seal::CKKSEncoder& encoder) {
    
    if (!aggressive_rescaling_) {
        return; // Skip rescaling if not using aggressive strategy
    }
    
    // Get the current context data
    auto context_data = context_->get_context_data(cipher.parms_id());
    if (!context_data) {
        return; // Cannot rescale further
    }
    
    // Check if we are at the last level
    if (context_data->chain_index() == context_->first_context_data()->chain_index()) {
        return; // At the first level, cannot rescale
    }
    
    // Check the scale - if it's significantly larger than intended, rescale
    if (cipher.scale() > scale_ * 1.2) {
        evaluator.rescale_to_next_inplace(cipher);
        cipher.scale() = scale_; // Reset to the intended scale
    }
}

// Main forward pass
seal::Ciphertext EncryptedTransformer::forward(
    const seal::Ciphertext& input_embedding,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator,
    seal::Ciphertext* attention_mask) {
    
    seal::Ciphertext current_output = input_embedding;
    
    // Process through multiple transformer layers
    for (int layer = 0; layer < num_layers_; layer++) {
        if (num_layers_ > 1) {
            std::cout << "Processing layer " << layer + 1 << "/" << num_layers_ << std::endl;
        }
        
        current_output = transformerLayer(
            current_output, encoder, encryptor, evaluator, attention_mask);
    }
    
    return current_output;
}

// Single transformer layer
seal::Ciphertext EncryptedTransformer::transformerLayer(
    const seal::Ciphertext& input,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator,
    seal::Ciphertext* attention_mask) {
    
    // Simplify for minimal baseline - just pass through
    if (hidden_size_ <= 8) {
        return input;
    }
    
    // Multi-head attention sublayer
    seal::Ciphertext attention_output = multiHeadAttention(
        input, encoder, encryptor, evaluator, attention_mask);
    
    // Add residual connection
    evaluator.add_inplace(attention_output, input);
    
    // Rescale after addition if using aggressive rescaling
    rescaleIfNeeded(attention_output, evaluator, encoder);
    
    // Feed-forward sublayer
    seal::Ciphertext ff_output = feedForward(
        attention_output, encoder, encryptor, evaluator);
    
    // Add residual connection
    evaluator.add_inplace(ff_output, attention_output);
    
    // Final rescale if needed
    rescaleIfNeeded(ff_output, evaluator, encoder);
    
    return ff_output;
}

// Multi-head attention
seal::Ciphertext EncryptedTransformer::multiHeadAttention(
    const seal::Ciphertext& input,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator,
    seal::Ciphertext* attention_mask) {
    
    // For minimal implementation, use only query and output weights
    seal::Ciphertext result;
    
    // Compute query projection
    evaluator.multiply_plain(input, weights_->getQueryWeight(), result);
    evaluator.relinearize_inplace(result, relin_keys_);
    
    // Rescale after multiplication
    if (aggressive_rescaling_) {
        evaluator.rescale_to_next_inplace(result);
        result.scale() = scale_;
    }
    
    // Compute output projection
    evaluator.multiply_plain(result, weights_->getOutputWeight(), result);
    evaluator.relinearize_inplace(result, relin_keys_);
    
    // Rescale after multiplication
    if (aggressive_rescaling_) {
        evaluator.rescale_to_next_inplace(result);
        result.scale() = scale_;
    }
    
    return result;
}

// Feed-forward network
seal::Ciphertext EncryptedTransformer::feedForward(
    const seal::Ciphertext& input,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator) {
    
    // For minimal implementation, use simple projection
    seal::Ciphertext result;
    
    // First dense layer
    evaluator.multiply_plain(input, weights_->getFeedForward1Weight(), result);
    evaluator.relinearize_inplace(result, relin_keys_);
    
    // Rescale after multiplication
    if (aggressive_rescaling_) {
        evaluator.rescale_to_next_inplace(result);
        result.scale() = scale_;
    }
    
    // Apply simplified activation (skip activation for minimal implementation)
    
    // Second dense layer
    evaluator.multiply_plain(result, weights_->getFeedForward2Weight(), result);
    evaluator.relinearize_inplace(result, relin_keys_);
    
    // Rescale after multiplication
    if (aggressive_rescaling_) {
        evaluator.rescale_to_next_inplace(result);
        result.scale() = scale_;
    }
    
    return result;
} 