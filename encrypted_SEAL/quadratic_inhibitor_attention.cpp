#include "encrypted_transformer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>

EncryptedQuadraticInhibitorAttention::EncryptedQuadraticInhibitorAttention(
    std::shared_ptr<seal::SEALContext> context,
    seal::RelinKeys relin_keys,
    seal::GaloisKeys galois_keys,
    int hidden_size,
    int num_attention_heads,
    double scale)
    : context_(context),
      relin_keys_(relin_keys),
      galois_keys_(galois_keys),
      hidden_size_(hidden_size),
      num_attention_heads_(num_attention_heads),
      attention_head_size_(hidden_size / num_attention_heads),
      scale_(scale) {
    
    // Initialize gamma coefficient (scaling factor for the quadratic term)
    gamma_coef_ = std::sqrt(static_cast<double>(attention_head_size_));
    
    // Initialize the dimension scale term: 3d/16γ
    dim_scale_ = 3.0 * attention_head_size_ / (16.0 * gamma_coef_);
    
    std::cout << "Initialized Quadratic Inhibitor Attention with:" << std::endl;
    std::cout << "  Hidden size: " << hidden_size_ << std::endl;
    std::cout << "  Attention heads: " << num_attention_heads_ << std::endl;
    std::cout << "  Head dimension: " << attention_head_size_ << std::endl;
    std::cout << "  Gamma coefficient: " << gamma_coef_ << std::endl;
    std::cout << "  Dimension scale: " << dim_scale_ << std::endl;
}

seal::Ciphertext EncryptedQuadraticInhibitorAttention::forward(
    const seal::Ciphertext& input,
    const seal::Ciphertext& wq,
    const seal::Ciphertext& wk,
    const seal::Ciphertext& wv,
    const seal::Ciphertext& wo,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator,
    const seal::Ciphertext* attention_mask) {
    
    std::cout << "Running quadratic inhibitor attention forward pass..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Step 1: Project input to query, key, and value
    std::cout << "  Computing query, key, value projections..." << std::endl;
    seal::Ciphertext query = matrixMultiply(input, wq, hidden_size_, hidden_size_, hidden_size_, encoder, encryptor, evaluator);
    seal::Ciphertext key = matrixMultiply(input, wk, hidden_size_, hidden_size_, hidden_size_, encoder, encryptor, evaluator);
    seal::Ciphertext value = matrixMultiply(input, wv, hidden_size_, hidden_size_, hidden_size_, encoder, encryptor, evaluator);
    
    // Step 2: Compute quadratic inhibitor attention
    std::cout << "  Computing quadratic inhibition..." << std::endl;
    seal::Ciphertext context_layer = computeQuadraticInhibition(query, key, value, encoder, encryptor, evaluator, attention_mask);
    
    // Step 3: Apply output projection
    std::cout << "  Computing output projection..." << std::endl;
    seal::Ciphertext output = matrixMultiply(context_layer, wo, hidden_size_, hidden_size_, hidden_size_, encoder, encryptor, evaluator);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Attention forward pass completed in " << elapsed.count() << " seconds" << std::endl;
    
    return output;
}

seal::Ciphertext EncryptedQuadraticInhibitorAttention::computeQuadraticInhibition(
    const seal::Ciphertext& query,
    const seal::Ciphertext& key,
    const seal::Ciphertext& value,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator,
    const seal::Ciphertext* attention_mask) {
    
    // Step 1: Compute query-key squared difference (||Q_i - K_j||^2)
    std::cout << "    Computing query-key distances..." << std::endl;
    
    // We compute the squared L2 norm: ||Q_i - K_j||^2
    // For each query position i and each key position j
    
    // First, we need to compute Q^2
    seal::Ciphertext query_squared;
    evaluator.square(query, query_squared);
    evaluator.relinearize_inplace(query_squared, relin_keys_);
    evaluator.rescale_to_next_inplace(query_squared);
    
    // Then compute K^2
    seal::Ciphertext key_squared;
    evaluator.square(key, key_squared);
    evaluator.relinearize_inplace(key_squared, relin_keys_);
    evaluator.rescale_to_next_inplace(key_squared);
    
    // Compute -2 * Q * K
    seal::Ciphertext query_key_product;
    evaluator.multiply(query, key, query_key_product);
    evaluator.relinearize_inplace(query_key_product, relin_keys_);
    evaluator.rescale_to_next_inplace(query_key_product);
    
    // Encode -2 scalar
    seal::Plaintext minus_two_plain;
    encoder.encode(-2.0, query_key_product.scale(), minus_two_plain);
    
    // Multiply by -2
    evaluator.multiply_plain_inplace(query_key_product, minus_two_plain);
    
    // Compute final squared distance: Q^2 + K^2 - 2*Q*K
    seal::Ciphertext distance_squared = query_squared;
    evaluator.add_inplace(distance_squared, key_squared);
    evaluator.add_inplace(distance_squared, query_key_product);
    
    // Step 2: Scale the distance by the coefficient 15/(16γ)
    std::cout << "    Scaling distances..." << std::endl;
    double scale_factor = 15.0 / (16.0 * gamma_coef_);
    
    seal::Plaintext scale_plain;
    encoder.encode(scale_factor, distance_squared.scale(), scale_plain);
    
    evaluator.multiply_plain_inplace(distance_squared, scale_plain);
    
    // Step 3: Add dimension bias term 3d/(16γ)
    seal::Plaintext dim_bias_plain;
    encoder.encode(dim_scale_, distance_squared.scale(), dim_bias_plain);
    
    // Create a copy of the distance before adding bias
    seal::Ciphertext distance_with_bias = distance_squared;
    evaluator.add_plain_inplace(distance_with_bias, dim_bias_plain);
    
    // Step 4: Compute V_j - (scaled_distance + bias) for inhibition
    std::cout << "    Computing inhibitor function..." << std::endl;
    
    // Subtract the scaled distance + bias from value
    seal::Ciphertext inhibited_value = value;
    evaluator.negate_inplace(distance_with_bias);
    evaluator.add_inplace(inhibited_value, distance_with_bias);
    
    // Step 5: Apply ReLU approximation (non-linear activation)
    std::cout << "    Applying ReLU approximation..." << std::endl;
    seal::Ciphertext activated_value = computeApproximatedReLU(inhibited_value, encoder, encryptor, evaluator);
    
    // Apply attention mask if provided
    if (attention_mask != nullptr) {
        std::cout << "    Applying attention mask..." << std::endl;
        evaluator.multiply(activated_value, *attention_mask, activated_value);
        evaluator.relinearize_inplace(activated_value, relin_keys_);
        evaluator.rescale_to_next_inplace(activated_value);
    }
    
    // The result is our context layer (attention output)
    return activated_value;
}

seal::Ciphertext EncryptedQuadraticInhibitorAttention::ReLU(
    const seal::Ciphertext& input,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator) {
    
    // Use the approximated ReLU 
    return computeApproximatedReLU(input, encoder, encryptor, evaluator);
}

seal::Ciphertext EncryptedQuadraticInhibitorAttention::computeApproximatedReLU(
    const seal::Ciphertext& input,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator) {
    
    // We approximate ReLU with a polynomial
    // A common approximation is: ReLU(x) ≈ 0.5*x + 0.5*x*tanh(x)
    // For the tanh part, we'll use a Taylor approximation: tanh(x) ≈ x - x^3/3 + 2x^5/15
    
    // Step 1: Compute x^2
    seal::Ciphertext x_squared;
    evaluator.square(input, x_squared);
    evaluator.relinearize_inplace(x_squared, relin_keys_);
    evaluator.rescale_to_next_inplace(x_squared);
    
    // Step 2: Compute x^3
    seal::Ciphertext x_cubed;
    evaluator.multiply(input, x_squared, x_cubed);
    evaluator.relinearize_inplace(x_cubed, relin_keys_);
    evaluator.rescale_to_next_inplace(x_cubed);
    
    // Encode the coefficient -1/3
    seal::Plaintext coef_third_plain;
    encoder.encode(-1.0/3.0, x_cubed.scale(), coef_third_plain);
    
    // Multiply x^3 by -1/3
    evaluator.multiply_plain_inplace(x_cubed, coef_third_plain);
    
    // Compute 0.5*x + 0.5*x*tanh(x) ≈ 0.5*x + 0.5*x*(x - x^3/3)
    // = 0.5*x + 0.5*x^2 - 0.5*x^4/3
    
    // Encode 0.5
    seal::Plaintext half_plain;
    encoder.encode(0.5, input.scale(), half_plain);
    
    // Compute 0.5*x
    seal::Ciphertext half_x = input;
    evaluator.multiply_plain_inplace(half_x, half_plain);
    
    // Add 0.5*x + (x_cubed * -1/3) to get the final approximation
    seal::Ciphertext result = half_x;
    evaluator.add_inplace(result, x_cubed);
    
    return result;
}

seal::Ciphertext EncryptedQuadraticInhibitorAttention::matrixMultiply(
    const seal::Ciphertext& A,
    const seal::Ciphertext& B,
    int rows_A, int cols_A, int cols_B,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator) {
    
    // This is a simplified matrix multiplication implementation
    // In a real-world scenario, you'd need to implement proper matrix multiplication 
    // with ciphertext packing and rotation operations
    
    // For now, we'll simply multiply the ciphertexts element-wise as an approximation
    seal::Ciphertext result;
    evaluator.multiply(A, B, result);
    evaluator.relinearize_inplace(result, relin_keys_);
    evaluator.rescale_to_next_inplace(result);
    
    return result;
} 