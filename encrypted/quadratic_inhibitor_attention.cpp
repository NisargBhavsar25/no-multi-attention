#include "encrypted_transformer.h"
#include <cmath>
#include <iostream>
#include <algorithm>

EncryptedQuadraticInhibitorAttention::EncryptedQuadraticInhibitorAttention(
    heongpu::HEContext<heongpu::Scheme::CKKS>& context,
    heongpu::Relinkey<heongpu::Scheme::CKKS>& relin_key,
    heongpu::Galoiskey<heongpu::Scheme::CKKS>& galois_key,
    int hidden_size,
    int num_attention_heads,
    double scale)
    : context_(context),
      operators_(context),
      relin_key_(relin_key),
      galois_key_(galois_key),
      hidden_size_(hidden_size),
      num_attention_heads_(num_attention_heads),
      attention_head_size_(hidden_size / num_attention_heads),
      scale_(scale) {
    
    // Initialize gamma coefficient for scaling the quadratic form
    // Similar to PyTorch implementation - use sqrt(attention_head_size) for better stability
    gamma_coef_ = std::sqrt(attention_head_size_);
    
    // Calculate dimension scale term (3d/16)
    dim_scale_ = 3.0 * attention_head_size_ / 16.0;
    
    std::cout << "Initialized EncryptedQuadraticInhibitorAttention with:" << std::endl;
    std::cout << "  Hidden size: " << hidden_size_ << std::endl;
    std::cout << "  Attention heads: " << num_attention_heads_ << std::endl;
    std::cout << "  Head size: " << attention_head_size_ << std::endl;
    std::cout << "  Gamma coefficient: " << gamma_coef_ << std::endl;
    std::cout << "  Dimension scale: " << dim_scale_ << std::endl;
}

heongpu::Ciphertext<heongpu::Scheme::CKKS> EncryptedQuadraticInhibitorAttention::forward(
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& wq,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& wk,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& wv,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& wo,
    heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
    heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor) {
    
    // Set execution options for GPU operations
    heongpu::ExecutionOptions options;
    options.set_storage_type(heongpu::storage_type::DEVICE)
           .set_initial_location(true);
    
    // Linear projections for Q, K, V
    heongpu::Ciphertext<heongpu::Scheme::CKKS> query_layer(context_);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> key_layer(context_);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> value_layer(context_);
    
    // Perform matrix multiplication for query, key, value projections
    // Note: In a real implementation, these would be matrix-matrix multiplications
    // For this example, we're simplifying with element-wise multiplication
    operators_.multiply(input, wq, query_layer, options);
    operators_.relinearize_inplace(query_layer, relin_key_);
    
    operators_.multiply(input, wk, key_layer, options);
    operators_.relinearize_inplace(key_layer, relin_key_);
    
    operators_.multiply(input, wv, value_layer, options);
    operators_.relinearize_inplace(value_layer, relin_key_);
    
    // Apply quadratic inhibitor attention
    heongpu::Ciphertext<heongpu::Scheme::CKKS> attention_output = 
        computeQuadraticInhibition(query_layer, key_layer, value_layer, encoder, encryptor);
    
    // Output projection
    heongpu::Ciphertext<heongpu::Scheme::CKKS> output(context_);
    operators_.multiply(attention_output, wo, output, options);
    operators_.relinearize_inplace(output, relin_key_);
    
    return output;
}

heongpu::Ciphertext<heongpu::Scheme::CKKS> EncryptedQuadraticInhibitorAttention::computeQuadraticInhibition(
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& query,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& key,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& value,
    heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
    heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor) {
    
    // Set execution options for GPU operations
    heongpu::ExecutionOptions options;
    options.set_storage_type(heongpu::storage_type::DEVICE)
           .set_initial_location(true);
    
    // 1. Compute difference: query - key
    heongpu::Ciphertext<heongpu::Scheme::CKKS> diff(context_);
    operators_.sub(query, key, diff, options);
    
    // 2. Square the difference: (query - key)²
    heongpu::Ciphertext<heongpu::Scheme::CKKS> diff_squared(context_);
    operators_.multiply(diff, diff, diff_squared, options);
    operators_.relinearize_inplace(diff_squared, relin_key_);
    
    // 3. Scale with coefficient: 15/(16*gamma) * (query - key)²
    // Encode the scaling coefficient
    double coef = 15.0 / (16.0 * gamma_coef_ * std::sqrt(attention_head_size_));
    std::vector<double> coef_vec(hidden_size_, coef);
    
    heongpu::Plaintext<heongpu::Scheme::CKKS> coef_plain(context_);
    encoder.encode(coef_plain, coef_vec, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> coef_cipher(context_);
    encryptor.encrypt(coef_cipher, coef_plain);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> scaled_diff_squared(context_);
    operators_.multiply(diff_squared, coef_cipher, scaled_diff_squared, options);
    operators_.relinearize_inplace(scaled_diff_squared, relin_key_);
    
    // 4. Add dimension term: 15/(16*gamma) * (query - key)² + 3d/(16*gamma)
    // Encode the dimension term
    double dim_term = dim_scale_ / (gamma_coef_ * std::sqrt(attention_head_size_));
    std::vector<double> dim_vec(hidden_size_, dim_term);
    
    heongpu::Plaintext<heongpu::Scheme::CKKS> dim_plain(context_);
    encoder.encode(dim_plain, dim_vec, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> dim_cipher(context_);
    encryptor.encrypt(dim_cipher, dim_plain);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> z_scores(context_);
    operators_.add(scaled_diff_squared, dim_cipher, z_scores, options);
    
    // 5. Create ciphertext with constant 1 for subtraction
    std::vector<double> one_vec(hidden_size_, 1.0);
    
    heongpu::Plaintext<heongpu::Scheme::CKKS> one_plain(context_);
    encoder.encode(one_plain, one_vec, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> one_cipher(context_);
    encryptor.encrypt(one_cipher, one_plain);
    
    // 6. Compute inhibition: 1 - (15/(16*gamma) * (query - key)² + 3d/(16*gamma))
    heongpu::Ciphertext<heongpu::Scheme::CKKS> inhibitor(context_);
    operators_.sub(one_cipher, z_scores, inhibitor, options);
    
    // 7. Approximate ReLU: max(0, inhibitor)
    // Note: ReLU is non-polynomial, so we use a polynomial approximation
    // For this example, we'll use a simple approximation
    // In a real implementation, you would use a more accurate polynomial approximation
    // Here we'll assume all values are positive for simplicity (not a real ReLU)
    
    // 8. Apply inhibition to values: value * inhibitor
    heongpu::Ciphertext<heongpu::Scheme::CKKS> inhibited_values(context_);
    operators_.multiply(value, inhibitor, inhibited_values, options);
    operators_.relinearize_inplace(inhibited_values, relin_key_);
    
    return inhibited_values;
} 