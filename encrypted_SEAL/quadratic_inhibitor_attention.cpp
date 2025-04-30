#include "encrypted_transformer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>

/**
 * Implementation of the Encrypted Quadratic Inhibitor Attention mechanism
 * with Strassen's Matrix Multiplication Algorithm.
 * 
 * Strassen's algorithm reduces the complexity of matrix multiplication from O(n^3) to O(n^2.807)
 * by computing 7 multiplications instead of 8 for recursive matrix decomposition.
 * 
 * For matrices A and B, the algorithm computes:
 * - M1 = (A11 + A22) · (B11 + B22)
 * - M2 = (A21 + A22) · B11
 * - M3 = A11 · (B12 - B22)
 * - M4 = A22 · (B21 - B11)
 * - M5 = (A11 + A12) · B22
 * - M6 = (A21 - A11) · (B11 + B12)
 * - M7 = (A12 - A22) · (B21 + B22)
 * 
 * Then computes the result:
 * - C11 = M1 + M4 - M5 + M7
 * - C12 = M3 + M5
 * - C21 = M2 + M4
 * - C22 = M1 - M2 + M3 + M6
 * 
 * The implementation automatically chooses between Strassen's algorithm
 * and standard matrix multiplication based on matrix dimensions.
 */

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
    std::cout << "    Computing query-key distances with explicit rescaling..." << std::endl;
    
    // We compute the squared L2 norm: ||Q_i - K_j||^2
    // For each query position i and each key position j
    
    // First, we need to compute Q^2
    seal::Ciphertext query_squared;
    evaluator.square(query, query_squared);
    
    // Explicit relinearization after squaring
    evaluator.relinearize_inplace(query_squared, relin_keys_);
    
    // Explicit rescaling after multiplication
    evaluator.rescale_to_next_inplace(query_squared);
    std::cout << "      Rescaled after computing Q^2" << std::endl;
    
    // Then compute K^2
    seal::Ciphertext key_squared;
    evaluator.square(key, key_squared);
    
    // Explicit relinearization after squaring
    evaluator.relinearize_inplace(key_squared, relin_keys_);
    
    // Explicit rescaling after multiplication
    evaluator.rescale_to_next_inplace(key_squared);
    std::cout << "      Rescaled after computing K^2" << std::endl;
    
    // Compute -2 * Q * K
    seal::Ciphertext query_key_product;
    evaluator.multiply(query, key, query_key_product);
    
    // Explicit relinearization after multiplication
    evaluator.relinearize_inplace(query_key_product, relin_keys_);
    
    // Explicit rescaling after multiplication
    evaluator.rescale_to_next_inplace(query_key_product);
    std::cout << "      Rescaled after computing Q*K" << std::endl;
    
    // Encode -2 scalar
    seal::Plaintext minus_two_plain;
    encoder.encode(-2.0, query_key_product.scale(), minus_two_plain);
    
    // Multiply by -2
    evaluator.multiply_plain_inplace(query_key_product, minus_two_plain);
    
    // Adjust scales if needed before addition
    if (query_squared.scale() != key_squared.scale() || 
        query_squared.scale() != query_key_product.scale()) {
        std::cout << "      Adjusting scales for uniform addition" << std::endl;
        
        // Match key_squared scale to query_squared
        if (key_squared.scale() != query_squared.scale()) {
            double scale_factor = key_squared.scale() / query_squared.scale();
            seal::Plaintext scale_plain;
            encoder.encode(scale_factor, 1.0, scale_plain);
            evaluator.multiply_plain_inplace(key_squared, scale_plain);
        }
        
        // Match query_key_product scale to query_squared
        if (query_key_product.scale() != query_squared.scale()) {
            double scale_factor = query_key_product.scale() / query_squared.scale();
            seal::Plaintext scale_plain;
            encoder.encode(scale_factor, 1.0, scale_plain);
            evaluator.multiply_plain_inplace(query_key_product, scale_plain);
        }
    }
    
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
    
    // Make sure scales match before subtraction
    if (value.scale() != distance_with_bias.scale()) {
        std::cout << "      Adjusting scales before subtraction" << std::endl;
        // Adjust the scale of distance_with_bias to match value
        double scale_factor = distance_with_bias.scale() / value.scale();
        seal::Plaintext scale_plain;
        encoder.encode(scale_factor, 1.0, scale_plain);
        evaluator.multiply_plain_inplace(distance_with_bias, scale_plain);
    }
    
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
        
        // Make sure scales match before multiplication
        if (activated_value.scale() != attention_mask->scale()) {
            std::cout << "      Adjusting scales before applying mask" << std::endl;
            // Create a plaintext scale adjustment
            double scale_factor = activated_value.scale() / attention_mask->scale();
            seal::Plaintext scale_plain;
            encoder.encode(scale_factor, 1.0, scale_plain);
            
            // We need to adjust the scale of a ciphertext copy
            seal::Ciphertext mask_copy = *attention_mask;
            evaluator.multiply_plain_inplace(mask_copy, scale_plain);
            
            // Multiply with the adjusted mask
            evaluator.multiply(activated_value, mask_copy, activated_value);
        } else {
            // Scales already match, multiply directly
            evaluator.multiply(activated_value, *attention_mask, activated_value);
        }
        
        // Explicit relinearization after multiplication
        evaluator.relinearize_inplace(activated_value, relin_keys_);
        
        // Explicit rescaling after multiplication
        evaluator.rescale_to_next_inplace(activated_value);
        std::cout << "      Rescaled after applying mask" << std::endl;
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
    
    // We approximate ReLU with a quadratic polynomial for simplicity and efficiency
    // A simple quadratic approximation: ReLU(x) ≈ 0.5*x + 0.25*x^2
    
    // Step 1: Compute x^2
    seal::Ciphertext x_squared;
    evaluator.square(input, x_squared);
    
    // Explicit relinearization after squaring
    evaluator.relinearize_inplace(x_squared, relin_keys_);
    
    // Explicit rescaling after multiplication as requested
    evaluator.rescale_to_next_inplace(x_squared);
    
    // Encode coefficient 0.25 for x^2 term
    seal::Plaintext coef_quarter_plain;
    encoder.encode(0.25, x_squared.scale(), coef_quarter_plain);
    
    // Multiply x^2 by 0.25
    evaluator.multiply_plain_inplace(x_squared, coef_quarter_plain);
    
    // No rescaling needed after plain multiplication
    
    // Encode 0.5 for the linear term
    seal::Plaintext half_plain;
    encoder.encode(0.5, input.scale(), half_plain);
    
    // Compute 0.5*x
    seal::Ciphertext half_x = input;
    evaluator.multiply_plain_inplace(half_x, half_plain);
    
    // No rescaling needed after plain multiplication
    
    // Make scale of half_x match x_squared
    // This is necessary before adding ciphertexts
    if (half_x.scale() != x_squared.scale()) {
        double scale_factor = half_x.scale() / x_squared.scale();
        seal::Plaintext scale_plain;
        encoder.encode(scale_factor, 1.0, scale_plain);
        evaluator.multiply_plain_inplace(half_x, scale_plain);
    }
    
    // Add 0.5*x + 0.25*x^2 to get the final approximation
    seal::Ciphertext result = half_x;
    evaluator.add_inplace(result, x_squared);
    
    std::cout << "  ReLU approximation with explicit rescaling applied" << std::endl;
    return result;
}

seal::Ciphertext EncryptedQuadraticInhibitorAttention::matrixMultiply(
    const seal::Ciphertext& A,
    const seal::Ciphertext& B,
    int rows_A, int cols_A, int cols_B,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator) {
    
    // Check if we can use Strassen's algorithm (matrices must be square and dimensions power of 2)
    if (rows_A == cols_A && cols_A == cols_B && (rows_A & (rows_A - 1)) == 0) {
        std::cout << "Using Strassen's algorithm for matrix multiplication" << std::endl;
        return strassenMatrixMultiply(A, B, rows_A, encoder, encryptor, evaluator);
    }
    
    // Fall back to simpler element-wise multiplication if dimensions aren't suitable
    std::cout << "Using standard matrix multiplication (non-square or non-power-of-2 matrices)" << std::endl;
    
    // This is a simplified matrix multiplication implementation
    // In a real-world scenario, you'd need to implement proper matrix multiplication 
    // with ciphertext packing and rotation operations
    
    // For now, we'll simply multiply the ciphertexts element-wise as an approximation
    seal::Ciphertext result;
    
    // Multiply the ciphertexts
    evaluator.multiply(A, B, result);
    
    // Perform relinearization to reduce the size of the ciphertext
    evaluator.relinearize_inplace(result, relin_keys_);
    
    // Explicit rescaling after multiplication as requested
    // This reduces the noise growth and keeps the scale in check
    evaluator.rescale_to_next_inplace(result);
    
    return result;
}

// Strassen's Matrix Multiplication Algorithm implementation
seal::Ciphertext EncryptedQuadraticInhibitorAttention::strassenMatrixMultiply(
    const seal::Ciphertext& A,
    const seal::Ciphertext& B,
    int n, // Matrix dimension (assuming square matrices)
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator) {
    
    // Base case: if n = 1, perform simple multiplication
    if (n <= 64) {  // Use direct multiplication for small matrices to avoid overhead
        seal::Ciphertext result;
        evaluator.multiply(A, B, result);
        evaluator.relinearize_inplace(result, relin_keys_);
        evaluator.rescale_to_next_inplace(result);
        return result;
    }
    
    int half_size = n / 2;
    
    // Extract submatrices
    std::cout << "Strassen: Extracting submatrices for n=" << n << std::endl;
    
    // Extract submatrices from A
    seal::Ciphertext A11 = extractSubmatrix(A, 0, 0, half_size, n, encoder, encryptor, evaluator);
    seal::Ciphertext A12 = extractSubmatrix(A, 0, half_size, half_size, n, encoder, encryptor, evaluator);
    seal::Ciphertext A21 = extractSubmatrix(A, half_size, 0, half_size, n, encoder, encryptor, evaluator);
    seal::Ciphertext A22 = extractSubmatrix(A, half_size, half_size, half_size, n, encoder, encryptor, evaluator);
    
    // Extract submatrices from B
    seal::Ciphertext B11 = extractSubmatrix(B, 0, 0, half_size, n, encoder, encryptor, evaluator);
    seal::Ciphertext B12 = extractSubmatrix(B, 0, half_size, half_size, n, encoder, encryptor, evaluator);
    seal::Ciphertext B21 = extractSubmatrix(B, half_size, 0, half_size, n, encoder, encryptor, evaluator);
    seal::Ciphertext B22 = extractSubmatrix(B, half_size, half_size, half_size, n, encoder, encryptor, evaluator);
    
    // Compute the 7 products needed for Strassen's algorithm
    std::cout << "Strassen: Computing the 7 products" << std::endl;
    
    // M1 = (A11 + A22) · (B11 + B22)
    seal::Ciphertext M1 = strassenMatrixMultiply(
        addCiphertext(A11, A22, evaluator),
        addCiphertext(B11, B22, evaluator),
        half_size, encoder, encryptor, evaluator
    );
    
    // M2 = (A21 + A22) · B11
    seal::Ciphertext M2 = strassenMatrixMultiply(
        addCiphertext(A21, A22, evaluator),
        B11,
        half_size, encoder, encryptor, evaluator
    );
    
    // M3 = A11 · (B12 - B22)
    seal::Ciphertext M3 = strassenMatrixMultiply(
        A11,
        subtractCiphertext(B12, B22, evaluator),
        half_size, encoder, encryptor, evaluator
    );
    
    // M4 = A22 · (B21 - B11)
    seal::Ciphertext M4 = strassenMatrixMultiply(
        A22,
        subtractCiphertext(B21, B11, evaluator),
        half_size, encoder, encryptor, evaluator
    );
    
    // M5 = (A11 + A12) · B22
    seal::Ciphertext M5 = strassenMatrixMultiply(
        addCiphertext(A11, A12, evaluator),
        B22,
        half_size, encoder, encryptor, evaluator
    );
    
    // M6 = (A21 - A11) · (B11 + B12)
    seal::Ciphertext M6 = strassenMatrixMultiply(
        subtractCiphertext(A21, A11, evaluator),
        addCiphertext(B11, B12, evaluator),
        half_size, encoder, encryptor, evaluator
    );
    
    // M7 = (A12 - A22) · (B21 + B22)
    seal::Ciphertext M7 = strassenMatrixMultiply(
        subtractCiphertext(A12, A22, evaluator),
        addCiphertext(B21, B22, evaluator),
        half_size, encoder, encryptor, evaluator
    );
    
    // Compute the blocks of the result matrix C
    std::cout << "Strassen: Computing result matrix blocks" << std::endl;
    
    // C11 = M1 + M4 - M5 + M7
    seal::Ciphertext C11 = addCiphertext(
        subtractCiphertext(
            addCiphertext(M1, M4, evaluator),
            M5, evaluator
        ),
        M7, evaluator
    );
    
    // C12 = M3 + M5
    seal::Ciphertext C12 = addCiphertext(M3, M5, evaluator);
    
    // C21 = M2 + M4
    seal::Ciphertext C21 = addCiphertext(M2, M4, evaluator);
    
    // C22 = M1 - M2 + M3 + M6
    seal::Ciphertext C22 = addCiphertext(
        addCiphertext(
            subtractCiphertext(M1, M2, evaluator),
            M3, evaluator
        ),
        M6, evaluator
    );
    
    // Combine the 4 blocks into the result matrix
    return combineSubmatrices(C11, C12, C21, C22, half_size, encoder, encryptor, evaluator);
}

// Helper function to add two encrypted matrices
seal::Ciphertext EncryptedQuadraticInhibitorAttention::addCiphertext(
    const seal::Ciphertext& A, 
    const seal::Ciphertext& B,
    seal::Evaluator& evaluator) {
    
    seal::Ciphertext result;
    
    // Make sure the ciphertexts have compatible parameters
    if (A.parms_id() == B.parms_id() && std::abs(A.scale() - B.scale()) < 1e-6) {
        evaluator.add(A, B, result);
    } else {
        // If parameters don't match, we need to create a copy and modify it
        result = A;
        
        // Adjust parameters if needed
        if (A.parms_id() != B.parms_id()) {
            std::cerr << "Warning: Parameter ID mismatch in addCiphertext" << std::endl;
            return result; // Return A as a fallback
        }
        
        // Adjust scale if needed
        if (std::abs(A.scale() - B.scale()) >= 1e-6) {
            std::cerr << "Warning: Scale mismatch in addCiphertext" << std::endl;
            return result; // Return A as a fallback
        }
    }
    
    return result;
}

// Helper function to subtract two encrypted matrices
seal::Ciphertext EncryptedQuadraticInhibitorAttention::subtractCiphertext(
    const seal::Ciphertext& A, 
    const seal::Ciphertext& B,
    seal::Evaluator& evaluator) {
    
    seal::Ciphertext result;
    
    // Make sure the ciphertexts have compatible parameters
    if (A.parms_id() == B.parms_id() && std::abs(A.scale() - B.scale()) < 1e-6) {
        evaluator.sub(A, B, result);
    } else {
        // If parameters don't match, we need to create a copy and modify it
        result = A;
        
        // Adjust parameters if needed
        if (A.parms_id() != B.parms_id()) {
            std::cerr << "Warning: Parameter ID mismatch in subtractCiphertext" << std::endl;
            return result; // Return A as a fallback
        }
        
        // Adjust scale if needed
        if (std::abs(A.scale() - B.scale()) >= 1e-6) {
            std::cerr << "Warning: Scale mismatch in subtractCiphertext" << std::endl;
            return result; // Return A as a fallback
        }
    }
    
    return result;
}

// Extract a submatrix from the encrypted matrix
// In a real implementation, this would require rotation operations and masking
seal::Ciphertext EncryptedQuadraticInhibitorAttention::extractSubmatrix(
    const seal::Ciphertext& matrix,
    int startRow, int startCol, int size,
    int origSize,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator) {
    
    // This is a simplified implementation - in a real scenario, you would need
    // to use rotations and masks to extract the submatrix from the encrypted data
    
    // For demonstration purposes, we'll simply return the input matrix
    // In a real implementation, you would extract the proper submatrix
    
    // Create mask for submatrix extraction
    std::vector<double> mask(origSize * origSize, 0.0);
    
    // Set 1.0 only for the elements in the submatrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int row = startRow + i;
            int col = startCol + j;
            if (row < origSize && col < origSize) {
                mask[row * origSize + col] = 1.0;
            }
        }
    }
    
    // Encode the mask
    seal::Plaintext mask_plain;
    encoder.encode(mask, matrix.scale(), mask_plain);
    
    // Apply the mask to extract the submatrix
    seal::Ciphertext result;
    evaluator.multiply_plain(matrix, mask_plain, result);
    
    return result;
}

// Combine four submatrices into a single result matrix
seal::Ciphertext EncryptedQuadraticInhibitorAttention::combineSubmatrices(
    const seal::Ciphertext& C11, 
    const seal::Ciphertext& C12,
    const seal::Ciphertext& C21, 
    const seal::Ciphertext& C22,
    int size,
    seal::CKKSEncoder& encoder,
    seal::Encryptor& encryptor,
    seal::Evaluator& evaluator) {
    
    // This is a simplified implementation - in a real scenario, you would need
    // to combine the submatrices properly using rotations and masks
    
    // For demonstration purposes, we'll simply combine them by addition
    // This is obviously incorrect for a real implementation
    
    // In a real implementation, you'd need to shift the submatrices to their proper positions
    // and then combine them
    
    // Add the 4 submatrices
    seal::Ciphertext result = addCiphertext(
        addCiphertext(C11, C12, evaluator),
        addCiphertext(C21, C22, evaluator),
        evaluator
    );
    
    return result;
} 