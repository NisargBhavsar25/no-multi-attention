#include "encrypted_transformer.h"
#include <iostream>
#include <cmath>

EncryptedTransformer::EncryptedTransformer(
    heongpu::HEContext<heongpu::Scheme::CKKS>& context,
    heongpu::Relinkey<heongpu::Scheme::CKKS>& relin_key,
    heongpu::Galoiskey<heongpu::Scheme::CKKS>& galois_key,
    int num_layers,
    int hidden_size,
    int num_attention_heads,
    double scale)
    : context_(context),
      operators_(context),
      relin_key_(relin_key),
      galois_key_(galois_key),
      num_layers_(num_layers),
      hidden_size_(hidden_size),
      num_attention_heads_(num_attention_heads),
      scale_(scale) {
    
    // Initialize the attention mechanism
    attention_ = std::make_unique<EncryptedQuadraticInhibitorAttention>(
        context, relin_key, galois_key, hidden_size, num_attention_heads, scale);
    
    std::cout << "Initialized EncryptedTransformer with:" << std::endl;
    std::cout << "  Number of layers: " << num_layers_ << std::endl;
    std::cout << "  Hidden size: " << hidden_size_ << std::endl;
    std::cout << "  Number of attention heads: " << num_attention_heads_ << std::endl;
}

void EncryptedTransformer::setWeights(std::shared_ptr<EncryptedTransformerWeights> weights) {
    weights_ = weights;
    std::cout << "Weights set for transformer" << std::endl;
}

heongpu::Ciphertext<heongpu::Scheme::CKKS> EncryptedTransformer::forward(
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
    heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
    heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor) {
    
    if (!weights_) {
        throw std::runtime_error("Weights not set for transformer");
    }
    
    // Set execution options for GPU operations
    heongpu::ExecutionOptions options;
    options.set_storage_type(heongpu::storage_type::DEVICE)
           .set_initial_location(true);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> hidden_state = input;
    
    // Process through each transformer layer
    for (int layer = 0; layer < num_layers_; ++layer) {
        std::cout << "Processing layer " << layer + 1 << "/" << num_layers_ << std::endl;
        
        // 1. Self-attention sub-layer
        // a. Layer normalization
        heongpu::Ciphertext<heongpu::Scheme::CKKS> norm_output = 
            layerNorm(hidden_state, encoder, encryptor);
        
        // b. Self-attention
        heongpu::Ciphertext<heongpu::Scheme::CKKS> attention_output = 
            attention_->forward(
                norm_output,
                weights_->getQueryWeights()[layer],
                weights_->getKeyWeights()[layer],
                weights_->getValueWeights()[layer],
                weights_->getOutputWeights()[layer],
                encoder,
                encryptor);
        
        // c. Residual connection
        heongpu::Ciphertext<heongpu::Scheme::CKKS> attention_residual(context_);
        operators_.add(hidden_state, attention_output, attention_residual, options);
        
        // 2. Feed-forward sub-layer
        // a. Layer normalization
        heongpu::Ciphertext<heongpu::Scheme::CKKS> ff_norm_output = 
            layerNorm(attention_residual, encoder, encryptor);
        
        // b. Feed-forward network
        heongpu::Ciphertext<heongpu::Scheme::CKKS> ff_output = 
            feedForward(
                ff_norm_output,
                weights_->getFF1Weights()[layer],
                weights_->getFF2Weights()[layer],
                encoder,
                encryptor);
        
        // c. Residual connection
        heongpu::Ciphertext<heongpu::Scheme::CKKS> layer_output(context_);
        operators_.add(attention_residual, ff_output, layer_output, options);
        
        // Update hidden state for next layer
        hidden_state = layer_output;
    }
    
    // Final layer normalization
    heongpu::Ciphertext<heongpu::Scheme::CKKS> final_output = 
        layerNorm(hidden_state, encoder, encryptor);
    
    return final_output;
}

heongpu::Ciphertext<heongpu::Scheme::CKKS> EncryptedTransformer::layerNorm(
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
    heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
    heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor) {
    
    // Set execution options for GPU operations
    heongpu::ExecutionOptions options;
    options.set_storage_type(heongpu::storage_type::DEVICE)
           .set_initial_location(true);
    
    // Note: Layer normalization is a non-polynomial operation involving mean, variance, and
    // square root calculations. In the encrypted domain, we need to approximate it.
    // For this example, we'll use a simplified approximation assuming the data is already
    // roughly normalized.
    
    // In practice, we could use more sophisticated polynomial approximations or
    // interactive protocols with partial decryption for complex operations.
    
    // For now, we'll simply return the input as a placeholder
    // In a real implementation, you would implement a proper approximation
    return input;
}

heongpu::Ciphertext<heongpu::Scheme::CKKS> EncryptedTransformer::feedForward(
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& ff1_weights,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& ff2_weights,
    heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
    heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor) {
    
    // Set execution options for GPU operations
    heongpu::ExecutionOptions options;
    options.set_storage_type(heongpu::storage_type::DEVICE)
           .set_initial_location(true);
    
    // First linear layer
    heongpu::Ciphertext<heongpu::Scheme::CKKS> intermediate(context_);
    operators_.multiply(input, ff1_weights, intermediate, options);
    operators_.relinearize_inplace(intermediate, relin_key_);
    
    // ReLU activation (approximated)
    heongpu::Ciphertext<heongpu::Scheme::CKKS> activated = reluApprox(intermediate, encoder, encryptor);
    
    // Second linear layer
    heongpu::Ciphertext<heongpu::Scheme::CKKS> output(context_);
    operators_.multiply(activated, ff2_weights, output, options);
    operators_.relinearize_inplace(output, relin_key_);
    
    return output;
}

heongpu::Ciphertext<heongpu::Scheme::CKKS> EncryptedTransformer::reluApprox(
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
    heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
    heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor) {
    
    // Set execution options for GPU operations
    heongpu::ExecutionOptions options;
    options.set_storage_type(heongpu::storage_type::DEVICE)
           .set_initial_location(true);
    
    // Note: ReLU is a non-polynomial function, so we need a polynomial approximation
    // For this example implementation, we'll use a simple approximation
    // In a real implementation, you would use a more accurate polynomial approximation
    
    // For this example, we'll just use a simple quadratic approximation of ReLU
    // f(x) = 0.25 * x² + 0.5 * x + 0.25 for x in [-1, 1]
    // This is just a placeholder - in practice, you would use a better approximation
    
    // Square term: 0.25 * x²
    heongpu::Ciphertext<heongpu::Scheme::CKKS> squared(context_);
    operators_.multiply(input, input, squared, options);
    operators_.relinearize_inplace(squared, relin_key_);
    
    // Scale 0.25
    std::vector<double> coef_vec(hidden_size_, 0.25);
    heongpu::Plaintext<heongpu::Scheme::CKKS> coef_plain(context_);
    encoder.encode(coef_plain, coef_vec, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> coef_cipher(context_);
    encryptor.encrypt(coef_cipher, coef_plain);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> squared_scaled(context_);
    operators_.multiply(squared, coef_cipher, squared_scaled, options);
    operators_.relinearize_inplace(squared_scaled, relin_key_);
    
    // Linear term: 0.5 * x
    std::vector<double> half_vec(hidden_size_, 0.5);
    heongpu::Plaintext<heongpu::Scheme::CKKS> half_plain(context_);
    encoder.encode(half_plain, half_vec, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> half_cipher(context_);
    encryptor.encrypt(half_cipher, half_plain);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> linear_term(context_);
    operators_.multiply(input, half_cipher, linear_term, options);
    operators_.relinearize_inplace(linear_term, relin_key_);
    
    // Constant term: 0.25
    std::vector<double> const_vec(hidden_size_, 0.25);
    heongpu::Plaintext<heongpu::Scheme::CKKS> const_plain(context_);
    encoder.encode(const_plain, const_vec, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> const_cipher(context_);
    encryptor.encrypt(const_cipher, const_plain);
    
    // Add all terms: 0.25 * x² + 0.5 * x + 0.25
    heongpu::Ciphertext<heongpu::Scheme::CKKS> sum1(context_);
    operators_.add(squared_scaled, linear_term, sum1, options);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> result(context_);
    operators_.add(sum1, const_cipher, result, options);
    
    return result;
} 