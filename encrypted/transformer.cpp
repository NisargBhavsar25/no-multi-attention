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
    heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor,
    const heongpu::Ciphertext<heongpu::Scheme::CKKS>* attention_mask) {
    
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
                encryptor,
                attention_mask);
        
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
    
    // In standard layer normalization, we compute:
    // LayerNorm(x) = γ * (x - mean) / sqrt(variance + ε) + β
    // where γ and β are learnable parameters.
    
    // For homomorphic encryption, we need a polynomial approximation since we can't directly
    // compute means, variances, or square roots on encrypted data.
    
    // We'll use a centered polynomial approximation based on Taylor expansion:
    // 
    // We can assume the input is already approximately centered at 0,
    // so we mainly need to normalize the scale. We'll use a polynomial approximation
    // that maps a range of [-a, a] to approximately [-1, 1].
    
    // 1. First we need to scale down the potentially large values
    // We'll use a fixed scaling factor that we expect will work for most inputs
    double scale_factor = 1.0 / std::sqrt(hidden_size_);
    std::vector<double> scale_vec(hidden_size_, scale_factor);
    
    heongpu::Plaintext<heongpu::Scheme::CKKS> scale_plain(context_);
    encoder.encode(scale_plain, scale_vec, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> scale_cipher(context_);
    encryptor.encrypt(scale_cipher, scale_plain);
    
    // Scale the input by the factor
    heongpu::Ciphertext<heongpu::Scheme::CKKS> scaled_input(context_);
    operators_.multiply(input, scale_cipher, scaled_input, options);
    operators_.relinearize_inplace(scaled_input, relin_key_);
    
    // 2. Apply a polynomial stabilizer that approximately maintains
    // the distribution but reduces extreme values
    //
    // We use a cubic approximation: f(x) = x - αx³
    // This dampens large values while preserving small ones
    
    // Compute x³
    heongpu::Ciphertext<heongpu::Scheme::CKKS> input_squared(context_);
    operators_.multiply(scaled_input, scaled_input, input_squared, options);
    operators_.relinearize_inplace(input_squared, relin_key_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> input_cubed(context_);
    operators_.multiply(input_squared, scaled_input, input_cubed, options);
    operators_.relinearize_inplace(input_cubed, relin_key_);
    
    // Scale the cubic term with α = 0.1
    // This dampens large values while keeping smaller values mostly intact
    double cubic_coef = 0.1;
    std::vector<double> cubic_vec(hidden_size_, cubic_coef);
    
    heongpu::Plaintext<heongpu::Scheme::CKKS> cubic_plain(context_);
    encoder.encode(cubic_plain, cubic_vec, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> cubic_cipher(context_);
    encryptor.encrypt(cubic_cipher, cubic_plain);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> scaled_cubic(context_);
    operators_.multiply(input_cubed, cubic_cipher, scaled_cubic, options);
    operators_.relinearize_inplace(scaled_cubic, relin_key_);
    
    // Subtract from the scaled input: x - αx³
    heongpu::Ciphertext<heongpu::Scheme::CKKS> stabilized(context_);
    operators_.sub(scaled_input, scaled_cubic, stabilized, options);
    
    // 3. Finally, apply a learnable scale and bias (gamma and beta in standard LayerNorm)
    // For simplicity, we'll use fixed values here (gamma=1, beta=0),
    // but in a real implementation these would be learned parameters
    
    // The result is already scaled appropriately with gamma=1, beta=0
    return stabilized;
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
    
    // First linear layer (matrix multiplication)
    // Assuming dimensions: input [batch_size, hidden_size], ff1_weights [hidden_size, ff_dim]
    // Result: [batch_size, ff_dim]
    // For simplicity, we assume batch_size=1 in this implementation
    int ff_dim = 4 * hidden_size_; // Typical dimension for FF layer is 4x hidden size
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> intermediate = 
        matrixMultiply(input, ff1_weights, 1, hidden_size_, ff_dim, encoder, encryptor);
    
    // ReLU activation (approximated)
    heongpu::Ciphertext<heongpu::Scheme::CKKS> activated = reluApprox(intermediate, encoder, encryptor);
    
    // Second linear layer (matrix multiplication)
    // Assuming dimensions: activated [batch_size, ff_dim], ff2_weights [ff_dim, hidden_size]
    // Result: [batch_size, hidden_size]
    heongpu::Ciphertext<heongpu::Scheme::CKKS> output =
        matrixMultiply(activated, ff2_weights, 1, ff_dim, hidden_size_, encoder, encryptor);
    
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
    // We'll use the approximation: ReLU(x) ≈ 0.625x² + 0.5x
    
    // Square term: 0.625 * x²
    heongpu::Ciphertext<heongpu::Scheme::CKKS> squared(context_);
    operators_.multiply(input, input, squared, options);
    operators_.relinearize_inplace(squared, relin_key_);
    
    // Scale 0.625
    std::vector<double> coef1_vec(hidden_size_, 0.625);
    heongpu::Plaintext<heongpu::Scheme::CKKS> coef1_plain(context_);
    encoder.encode(coef1_plain, coef1_vec, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> coef1_cipher(context_);
    encryptor.encrypt(coef1_cipher, coef1_plain);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> squared_scaled(context_);
    operators_.multiply(squared, coef1_cipher, squared_scaled, options);
    operators_.relinearize_inplace(squared_scaled, relin_key_);
    
    // Linear term: 0.5 * x
    std::vector<double> coef2_vec(hidden_size_, 0.5);
    heongpu::Plaintext<heongpu::Scheme::CKKS> coef2_plain(context_);
    encoder.encode(coef2_plain, coef2_vec, scale_);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> coef2_cipher(context_);
    encryptor.encrypt(coef2_cipher, coef2_plain);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> linear_term(context_);
    operators_.multiply(input, coef2_cipher, linear_term, options);
    operators_.relinearize_inplace(linear_term, relin_key_);
    
    // Add terms: 0.625 * x² + 0.5 * x
    heongpu::Ciphertext<heongpu::Scheme::CKKS> result(context_);
    operators_.add(squared_scaled, linear_term, result, options);
    
    return result;
}

heongpu::Ciphertext<heongpu::Scheme::CKKS> EncryptedTransformer::matrixMultiply(
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
                operators_.multiply(A, a_mask_cipher, a_element, options);
                operators_.relinearize_inplace(a_element, relin_key_);
                
                heongpu::Ciphertext<heongpu::Scheme::CKKS> b_element(context_);
                operators_.multiply(B, b_mask_cipher, b_element, options);
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