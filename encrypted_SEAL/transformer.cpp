#include "encrypted_transformer.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <cmath>

EncryptedTransformer::EncryptedTransformer(
    std::shared_ptr<seal::SEALContext> context,
    seal::RelinKeys relin_keys,
    seal::GaloisKeys galois_keys,
    int num_layers,
    int hidden_size,
    int num_attention_heads,
    double scale)
    : context_(context),
      relin_keys_(relin_keys),
      galois_keys_(galois_keys),
      num_layers_(num_layers),
      hidden_size_(hidden_size),
      num_attention_heads_(num_attention_heads),
      scale_(scale) {
    
    // Create the attention mechanism
    attention_ = std::make_unique<EncryptedQuadraticInhibitorAttention>(
        context_, relin_keys_, galois_keys_, hidden_size_, num_attention_heads_, scale_);
    
    std::cout << "Initialized Encrypted Transformer with " << num_layers_ << " layers" << std::endl;
}

void EncryptedTransformer::setWeights(std::shared_ptr<EncryptedTransformerWeights> weights) {
    weights_ = weights;
    std::cout << "Weights set successfully" << std::endl;
}

seal::Ciphertext EncryptedTransformer::forward(
    const seal::Ciphertext& input,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator,
    const seal::Ciphertext* attention_mask) {
    
    if (!weights_) {
        throw std::runtime_error("Weights not set. Call setWeights() before forward().");
    }
    
    std::cout << "Running encrypted transformer forward pass with " << num_layers_ << " layers" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Start with the input
    seal::Ciphertext hidden_state = input;
    
    // Process through each transformer layer
    for (int layer = 0; layer < num_layers_; layer++) {
        std::cout << "Processing layer " << (layer + 1) << " of " << num_layers_ << std::endl;
        
        // Step 1: Self-attention
        auto attn_start = std::chrono::high_resolution_clock::now();
        
        // Get the weights for the current layer
        const auto& query_weight = weights_->getQueryWeights()[layer];
        const auto& key_weight = weights_->getKeyWeights()[layer];
        const auto& value_weight = weights_->getValueWeights()[layer];
        const auto& output_weight = weights_->getOutputWeights()[layer];
        
        // Get attention output
        seal::Ciphertext attention_output = attention_->forward(
            hidden_state, query_weight, key_weight, value_weight, output_weight,
            encoder, encryptor, evaluator, attention_mask);
        
        auto attn_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> attn_time = attn_end - attn_start;
        std::cout << "  Attention completed in " << attn_time.count() << " seconds" << std::endl;
        
        // Step 2: Add & Layer Norm (attention residual connection)
        evaluator.add_inplace(attention_output, hidden_state);
        
        // Apply layer normalization (approximated in encrypted domain)
        attention_output = layerNorm(attention_output, encoder, encryptor, evaluator);
        
        // Step 3: Feed-forward network
        auto ffn_start = std::chrono::high_resolution_clock::now();
        
        const auto& ff1_weight = weights_->getFF1Weights()[layer];
        const auto& ff2_weight = weights_->getFF2Weights()[layer];
        
        seal::Ciphertext ffn_output = feedForward(attention_output, ff1_weight, ff2_weight, encoder, encryptor, evaluator);
        
        auto ffn_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> ffn_time = ffn_end - ffn_start;
        std::cout << "  Feed-forward completed in " << ffn_time.count() << " seconds" << std::endl;
        
        // Step 4: Add & Layer Norm (FFN residual connection)
        evaluator.add_inplace(ffn_output, attention_output);
        
        // Apply layer normalization (approximated in encrypted domain)
        hidden_state = layerNorm(ffn_output, encoder, encryptor, evaluator);
        
        std::cout << "Layer " << (layer + 1) << " completed" << std::endl;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Transformer forward pass completed in " << elapsed.count() << " seconds" << std::endl;
    
    return hidden_state;
}

seal::Ciphertext EncryptedTransformer::layerNorm(
    const seal::Ciphertext& input,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator) {
    
    // In the encrypted domain, exact layer normalization is difficult
    // We'll implement a simplified version that approximates the effect
    
    // For now, we'll simply return the input since proper layer norm
    // would require computing mean and variance which involves rotations
    return input;
}

seal::Ciphertext EncryptedTransformer::feedForward(
    const seal::Ciphertext& input,
    const seal::Ciphertext& ff1_weights,
    const seal::Ciphertext& ff2_weights,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator) {
    
    // Step 1: First linear layer
    seal::Ciphertext intermediate = matrixMultiply(
        input, ff1_weights, hidden_size_, hidden_size_, 4 * hidden_size_, encoder, encryptor, evaluator);
    
    // Step 2: Apply GELU or ReLU activation
    intermediate = reluApprox(intermediate, encoder, encryptor, evaluator);
    
    // Step 3: Second linear layer
    seal::Ciphertext output = matrixMultiply(
        intermediate, ff2_weights, 4 * hidden_size_, 4 * hidden_size_, hidden_size_, encoder, encryptor, evaluator);
    
    return output;
}

seal::Ciphertext EncryptedTransformer::reluApprox(
    const seal::Ciphertext& input,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator) {
    
    // Delegate to the attention's ReLU implementation
    return attention_->ReLU(input, encoder, encryptor, evaluator);
}

seal::Ciphertext EncryptedTransformer::matrixMultiply(
    const seal::Ciphertext& A,
    const seal::Ciphertext& B,
    int rows_A, int cols_A, int cols_B,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator) {
    
    // Delegate to the attention's matrix multiply implementation
    return attention_->matrixMultiply(A, B, rows_A, cols_A, cols_B, encoder, encryptor, evaluator);
} 