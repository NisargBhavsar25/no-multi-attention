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
      encoder_(context),
      operators_(context, encoder_),
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

heongpu::Ciphertext<heongpu::Scheme::CKKS> EncryptedQuadraticInhibitorAttention::matrixMultiply(
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& A,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& B,
    int rows_A, int cols_A, int cols_B,
    heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
    heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor) {
    
    // Set execution options for GPU operations
    heongpu::ExecutionOptions options;
    options.set_storage_type(heongpu::storage_type::DEVICE)
           .set_initial_location(true);
    
    // Initialize result matrix as zeros
    std::vector<double> zeros(rows_A * cols_B, 0.0);
    heongpu::Plaintext<heongpu::Scheme::CKKS> zeros_plain(context_);
    encoder.encode(zeros_plain, zeros, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> result(context_);
    encryptor.encrypt(result, zeros_plain);
    
    // For each element in the output matrix
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            
            // Create mask for the current result position
            std::vector<double> result_mask(rows_A * cols_B, 0.0);
            result_mask[i * cols_B + j] = 1.0;
            
            heongpu::Plaintext<heongpu::Scheme::CKKS> result_mask_plain(context_);
            encoder.encode(result_mask_plain, result_mask, scale_);
            
            heongpu::Ciphertext<heongpu::Scheme::CKKS> result_mask_cipher(context_);
            encryptor.encrypt(result_mask_cipher, result_mask_plain);
            
            // For each element in the dot product
            heongpu::Ciphertext<heongpu::Scheme::CKKS> dot_product(context_);
            encryptor.encrypt(dot_product, zeros_plain); // Initialize to zeros
            
            for (int k = 0; k < cols_A; ++k) {
                // Create masks for accessing elements from A and B
                std::vector<double> a_mask(rows_A * cols_A, 0.0);
                a_mask[i * cols_A + k] = 1.0;
                
                std::vector<double> b_mask(cols_A * cols_B, 0.0);
                b_mask[k * cols_B + j] = 1.0;
                
                heongpu::Plaintext<heongpu::Scheme::CKKS> a_mask_plain(context_);
                encoder.encode(a_mask_plain, a_mask, scale_);
                
                heongpu::Plaintext<heongpu::Scheme::CKKS> b_mask_plain(context_);
                encoder.encode(b_mask_plain, b_mask, scale_);
                
                heongpu::Ciphertext<heongpu::Scheme::CKKS> a_mask_cipher(context_);
                encryptor.encrypt(a_mask_cipher, a_mask_plain);
                
                heongpu::Ciphertext<heongpu::Scheme::CKKS> b_mask_cipher(context_);
                encryptor.encrypt(b_mask_cipher, b_mask_plain);
                
                // Extract elements from A and B
                heongpu::Ciphertext<heongpu::Scheme::CKKS> a_element(context_);
                // Create a mutable copy of A
                heongpu::Ciphertext<heongpu::Scheme::CKKS> A_copy = A;
                operators_.multiply(A_copy, a_mask_cipher, a_element, options);
                operators_.relinearize_inplace(a_element, relin_key_);
                
                heongpu::Ciphertext<heongpu::Scheme::CKKS> b_element(context_);
                // Create a mutable copy of B
                heongpu::Ciphertext<heongpu::Scheme::CKKS> B_copy = B;
                operators_.multiply(B_copy, b_mask_cipher, b_element, options);
                operators_.relinearize_inplace(b_element, relin_key_);
                
                // Multiply and add to dot product
                heongpu::Ciphertext<heongpu::Scheme::CKKS> product(context_);
                operators_.multiply(a_element, b_element, product, options);
                operators_.relinearize_inplace(product, relin_key_);
                
                heongpu::Ciphertext<heongpu::Scheme::CKKS> updated_dot_product(context_);
                operators_.add(dot_product, product, updated_dot_product, options);
                dot_product = updated_dot_product;
            }
            
            // Update the result with the computed dot product
            heongpu::Ciphertext<heongpu::Scheme::CKKS> masked_dot_product(context_);
            operators_.multiply(dot_product, result_mask_cipher, masked_dot_product, options);
            operators_.relinearize_inplace(masked_dot_product, relin_key_);
            
            heongpu::Ciphertext<heongpu::Scheme::CKKS> updated_result(context_);
            operators_.add(result, masked_dot_product, updated_result, options);
            result = updated_result;
        }
    }
    
    return result;
}

heongpu::Ciphertext<heongpu::Scheme::CKKS> EncryptedQuadraticInhibitorAttention::forward(
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& wq,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& wk,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& wv,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& wo,
    heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
    heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>* attention_mask) {
    
    // Set execution options for GPU operations
    heongpu::ExecutionOptions options;
    options.set_storage_type(heongpu::storage_type::DEVICE)
           .set_initial_location(true);
    
    // Linear projections for Q, K, V using matrix multiplication
    // Assuming input dimensions: [batch_size, seq_len, hidden_size]
    // For simplicity, we assume batch_size=1, seq_len=1 in this implementation
    
    // Query projection: [1, hidden_size] x [hidden_size, hidden_size] -> [1, hidden_size]
    heongpu::Ciphertext<heongpu::Scheme::CKKS> query_layer = 
        matrixMultiply(input, wq, 1, hidden_size_, hidden_size_, encoder, encryptor);
    
    // Key projection: [1, hidden_size] x [hidden_size, hidden_size] -> [1, hidden_size]
    heongpu::Ciphertext<heongpu::Scheme::CKKS> key_layer = 
        matrixMultiply(input, wk, 1, hidden_size_, hidden_size_, encoder, encryptor);
    
    // Value projection: [1, hidden_size] x [hidden_size, hidden_size] -> [1, hidden_size]
    heongpu::Ciphertext<heongpu::Scheme::CKKS> value_layer = 
        matrixMultiply(input, wv, 1, hidden_size_, hidden_size_, encoder, encryptor);
    
    // Process multi-head attention
    // Instead of physically reshaping like in the PyTorch version, we'll use 
    // logical partitioning by processing heads one at a time and then combining results
    
    // Initialize a ciphertext to accumulate results across heads
    heongpu::Ciphertext<heongpu::Scheme::CKKS> attention_output(context_);
    
    // Create a zero vector for initialization
    std::vector<double> zeros(hidden_size_, 0.0);
    heongpu::Plaintext<heongpu::Scheme::CKKS> zero_plain(context_);
    encoder.encode(zero_plain, zeros, scale_);
    encryptor.encrypt(attention_output, zero_plain);
    
    // Process each attention head
    for (int head = 0; head < num_attention_heads_; ++head) {
        // Extract head-specific sections from Q, K, V
        // Since we can't physically reshape in HE, we'll extract sections via masks/rotations
        
        // Create masks for the current head
        std::vector<double> head_mask(hidden_size_, 0.0);
        int start_idx = head * attention_head_size_;
        int end_idx = start_idx + attention_head_size_;
        
        // Set mask to 1.0 for the current head's positions
        for (int i = start_idx; i < end_idx; ++i) {
            head_mask[i] = 1.0;
        }
        
        // Encode mask
        heongpu::Plaintext<heongpu::Scheme::CKKS> mask_plain(context_);
        encoder.encode(mask_plain, head_mask, scale_);
        
        heongpu::Ciphertext<heongpu::Scheme::CKKS> mask_cipher(context_);
        encryptor.encrypt(mask_cipher, mask_plain);
        
        // Apply mask to extract head-specific data
        heongpu::Ciphertext<heongpu::Scheme::CKKS> query_head(context_);
        heongpu::Ciphertext<heongpu::Scheme::CKKS> key_head(context_);
        heongpu::Ciphertext<heongpu::Scheme::CKKS> value_head(context_);
        
        operators_.multiply(query_layer, mask_cipher, query_head, options);
        operators_.relinearize_inplace(query_head, relin_key_);
    
        operators_.multiply(key_layer, mask_cipher, key_head, options);
        operators_.relinearize_inplace(key_head, relin_key_);
    
        operators_.multiply(value_layer, mask_cipher, value_head, options);
        operators_.relinearize_inplace(value_head, relin_key_);
    
        // Apply quadratic inhibitor attention to this head
        heongpu::Ciphertext<heongpu::Scheme::CKKS> head_output = 
            computeQuadraticInhibition(query_head, key_head, value_head, encoder, encryptor, attention_mask);
    
        // Accumulate head output 
        heongpu::Ciphertext<heongpu::Scheme::CKKS> temp_output(context_);
        operators_.add(attention_output, head_output, temp_output, options);
        attention_output = temp_output;
    }
    
    // Output projection using matrix multiplication
    // [1, hidden_size] x [hidden_size, hidden_size] -> [1, hidden_size]
    heongpu::Ciphertext<heongpu::Scheme::CKKS> output = 
        matrixMultiply(attention_output, wo, 1, hidden_size_, hidden_size_, encoder, encryptor);
    
    return output;
}

heongpu::Ciphertext<heongpu::Scheme::CKKS> EncryptedQuadraticInhibitorAttention::computeQuadraticInhibition(
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& query,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& key,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& value,
    heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
    heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>* attention_mask) {
    
    // Set execution options for GPU operations
    heongpu::ExecutionOptions options;
    options.set_storage_type(heongpu::storage_type::DEVICE)
           .set_initial_location(true);
    
    // Calculate attention scores: [1, hidden_size] x [hidden_size, 1] -> [1, 1]
    // Transpose of key for matrix multiplication
    
    // For simplicity, we assume we're dealing with a single query and key vector,
    // so the attention score is just the dot product of query and key
    std::vector<double> zeros(1, 0.0);
    heongpu::Plaintext<heongpu::Scheme::CKKS> zeros_plain(context_);
    encoder.encode(zeros_plain, zeros, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> attention_score(context_);
    encryptor.encrypt(attention_score, zeros_plain);
    
    // Compute dot product manually since we don't have a transpose operation
    for (int i = 0; i < attention_head_size_; ++i) {
        // Create masks to extract corresponding elements from query and key
        std::vector<double> query_mask(hidden_size_, 0.0);
        std::vector<double> key_mask(hidden_size_, 0.0);
        
        // Set mask to extract the i-th element
        int head_offset = 0; // We're assuming we're already working with head-specific data
        query_mask[head_offset + i] = 1.0;
        key_mask[head_offset + i] = 1.0;
    
        // Encode masks
        heongpu::Plaintext<heongpu::Scheme::CKKS> query_mask_plain(context_);
        encoder.encode(query_mask_plain, query_mask, scale_);
        
        heongpu::Plaintext<heongpu::Scheme::CKKS> key_mask_plain(context_);
        encoder.encode(key_mask_plain, key_mask, scale_);
        
        heongpu::Ciphertext<heongpu::Scheme::CKKS> query_mask_cipher(context_);
        encryptor.encrypt(query_mask_cipher, query_mask_plain);
        
        heongpu::Ciphertext<heongpu::Scheme::CKKS> key_mask_cipher(context_);
        encryptor.encrypt(key_mask_cipher, key_mask_plain);
        
        // Extract elements
        heongpu::Ciphertext<heongpu::Scheme::CKKS> query_element(context_);
        // Create a mutable copy of the query and query_mask_cipher
        heongpu::Ciphertext<heongpu::Scheme::CKKS> query_copy = query;
        heongpu::Ciphertext<heongpu::Scheme::CKKS> query_mask_cipher_copy = query_mask_cipher;
        operators_.multiply(query_copy, query_mask_cipher_copy, query_element, options);
        
        heongpu::Ciphertext<heongpu::Scheme::CKKS> key_element(context_);
        // Create a mutable copy of the key and key_mask_cipher
        heongpu::Ciphertext<heongpu::Scheme::CKKS> key_copy = key;
        heongpu::Ciphertext<heongpu::Scheme::CKKS> key_mask_cipher_copy = key_mask_cipher;
        operators_.multiply(key_copy, key_mask_cipher_copy, key_element, options);
        
        // Multiply elements and add to attention score
        heongpu::Ciphertext<heongpu::Scheme::CKKS> product(context_);
        operators_.multiply(query_element, key_element, product, options);
        operators_.relinearize_inplace(product, relin_key_);
        
        heongpu::Ciphertext<heongpu::Scheme::CKKS> updated_score(context_);
        operators_.add(attention_score, product, updated_score, options);
        attention_score = updated_score;
    }
    
    // Scale attention scores
    double scaling_factor = 1.0 / std::sqrt(static_cast<double>(attention_head_size_));
    std::vector<double> scale_vector(1, scaling_factor);
    heongpu::Plaintext<heongpu::Scheme::CKKS> scale_plain(context_);
    encoder.encode(scale_plain, scale_vector, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> scale_cipher(context_);
    encryptor.encrypt(scale_cipher, scale_plain);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> scaled_attention_score(context_);
    operators_.multiply(attention_score, scale_cipher, scaled_attention_score, options);
    operators_.relinearize_inplace(scaled_attention_score, relin_key_);
    
    // Apply attention mask if provided
    if (attention_mask != nullptr) {
        heongpu::Ciphertext<heongpu::Scheme::CKKS> masked_score(context_);
        // Create a mutable copy of the attention_mask
        heongpu::Ciphertext<heongpu::Scheme::CKKS> attention_mask_copy = *attention_mask;
        operators_.add(scaled_attention_score, attention_mask_copy, masked_score, options);
        scaled_attention_score = masked_score;
    }
    
    // Apply quadratic inhibition: ReLU(x)^2 - 0.5 * ReLU(x)^4
    // Compute ReLU approximation
    heongpu::Ciphertext<heongpu::Scheme::CKKS> relu_output = 
        computeApproximatedReLU(scaled_attention_score, encoder, encryptor);
    
    // Square the ReLU output
    heongpu::Ciphertext<heongpu::Scheme::CKKS> relu_squared(context_);
    operators_.multiply(relu_output, relu_output, relu_squared, options);
    operators_.relinearize_inplace(relu_squared, relin_key_);
    
    // Square again to get ReLU^4
    heongpu::Ciphertext<heongpu::Scheme::CKKS> relu_fourth(context_);
    operators_.multiply(relu_squared, relu_squared, relu_fourth, options);
    operators_.relinearize_inplace(relu_fourth, relin_key_);
    
    // Scale ReLU^4 by 0.5
    std::vector<double> half_scale(1, 0.5);
    heongpu::Plaintext<heongpu::Scheme::CKKS> half_scale_plain(context_);
    encoder.encode(half_scale_plain, half_scale, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> half_scale_cipher(context_);
    encryptor.encrypt(half_scale_cipher, half_scale_plain);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> half_relu_fourth(context_);
    operators_.multiply(relu_fourth, half_scale_cipher, half_relu_fourth, options);
    operators_.relinearize_inplace(half_relu_fourth, relin_key_);
    
    // Modified subtract operation using negate and add
    heongpu::Ciphertext<heongpu::Scheme::CKKS> negated_half_relu_fourth(context_);
    operators_.negate(half_relu_fourth, negated_half_relu_fourth, options);
    
    // Add missing declaration for quadratic_inhibition
    heongpu::Ciphertext<heongpu::Scheme::CKKS> quadratic_inhibition(context_);
    operators_.add(relu_squared, negated_half_relu_fourth, quadratic_inhibition, options);
    
    // Apply attention weights to value
    // For simplicity, we broadcast the quadratic_inhibition value (which is a scalar)
    // to all elements of the value vector
    
    // Create a vector of 1s for broadcasting
    std::vector<double> ones(hidden_size_, 1.0);
    heongpu::Plaintext<heongpu::Scheme::CKKS> ones_plain(context_);
    encoder.encode(ones_plain, ones, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> ones_cipher(context_);
    encryptor.encrypt(ones_cipher, ones_plain);
    
    // Broadcast the attention score
    heongpu::Ciphertext<heongpu::Scheme::CKKS> broadcasted_score(context_);
    operators_.multiply(ones_cipher, quadratic_inhibition, broadcasted_score, options);
    operators_.relinearize_inplace(broadcasted_score, relin_key_);
    
    // Multiply with value
    heongpu::Ciphertext<heongpu::Scheme::CKKS> context_layer(context_);
    // Create a mutable copy of the value
    heongpu::Ciphertext<heongpu::Scheme::CKKS> value_copy = value;
    operators_.multiply(broadcasted_score, value_copy, context_layer, options);
    operators_.relinearize_inplace(context_layer, relin_key_);
    
    return context_layer;
}

heongpu::Ciphertext<heongpu::Scheme::CKKS> EncryptedQuadraticInhibitorAttention::computeApproximatedReLU(
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
    heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
    heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor) {
    
    // Set execution options for GPU operations
    heongpu::ExecutionOptions options;
    options.set_storage_type(heongpu::storage_type::DEVICE)
           .set_initial_location(true);
    
    // ReLU approximation: 0.5x + 0.5x^2
    
    // First, compute x²
    heongpu::Ciphertext<heongpu::Scheme::CKKS> input_squared(context_);
    // Create a mutable copy of input
    heongpu::Ciphertext<heongpu::Scheme::CKKS> input_copy1 = input;
    operators_.multiply(input_copy1, input_copy1, input_squared, options);
    operators_.relinearize_inplace(input_squared, relin_key_);
    
    // Encode the coefficient 0.5
    std::vector<double> half_vec(hidden_size_, 0.5);
    heongpu::Plaintext<heongpu::Scheme::CKKS> half_plain(context_);
    encoder.encode(half_plain, half_vec, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> half_cipher(context_);
    encryptor.encrypt(half_cipher, half_plain);
    
    // Compute 0.5x²
    heongpu::Ciphertext<heongpu::Scheme::CKKS> relu_term1(context_);
    operators_.multiply(input_squared, half_cipher, relu_term1, options);
    operators_.relinearize_inplace(relu_term1, relin_key_);
    
    // Compute 0.5x
    heongpu::Ciphertext<heongpu::Scheme::CKKS> relu_term2(context_);
    // Create another mutable copy of input
    heongpu::Ciphertext<heongpu::Scheme::CKKS> input_copy2 = input;
    operators_.multiply(input_copy2, half_cipher, relu_term2, options);
    operators_.relinearize_inplace(relu_term2, relin_key_);
    
    // Combine terms to get 0.5x + 0.5x²
    heongpu::Ciphertext<heongpu::Scheme::CKKS> relu_output(context_);
    operators_.add(relu_term1, relu_term2, relu_output, options);
    
    return relu_output;
} 