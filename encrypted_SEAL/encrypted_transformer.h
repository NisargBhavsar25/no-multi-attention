#pragma once

#include <seal/seal.h>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

/**
 * Encrypted Quadratic Inhibitor Transformer Implementation
 * 
 * This implements a transformer with quadratic inhibitor attention mechanism
 * using the Microsoft SEAL library for homomorphic encryption operations.
 */

class EncryptedTransformerWeights {
public:
    EncryptedTransformerWeights(std::shared_ptr<seal::SEALContext> context);

    // Load and encrypt weights from pretrained model
    void loadFromPretrained(
        const std::string& model_path,
        seal::Encryptor& encryptor,
        seal::CKKSEncoder& encoder,
        double scale);

    // Add method to create dummy weights for testing
    void createDummyWeights(int hidden_size,
                           seal::Encryptor& encryptor,
                           seal::CKKSEncoder& encoder,
                           double scale);

    // Add method to print parameters for debugging
    void printParameterInfo() const {
        std::cout << "Debug: Weight parameter information" << std::endl;
        std::cout << "Query weight scale: " << query_weight_.scale() << std::endl;
        std::cout << "Key weight scale: " << key_weight_.scale() << std::endl;
        std::cout << "Value weight scale: " << value_weight_.scale() << std::endl;
        std::cout << "Output weight scale: " << output_weight_.scale() << std::endl;
        std::cout << "FF1 weight scale: " << ff1_weight_.scale() << std::endl;
        std::cout << "FF2 weight scale: " << ff2_weight_.scale() << std::endl;
    }

    // Access methods for weights
    seal::Plaintext& getQueryWeight() { return query_weight_; }
    seal::Plaintext& getKeyWeight() { return key_weight_; }
    seal::Plaintext& getValueWeight() { return value_weight_; }
    seal::Plaintext& getOutputWeight() { return output_weight_; }
    seal::Plaintext& getFeedForward1Weight() { return ff1_weight_; }
    seal::Plaintext& getFeedForward2Weight() { return ff2_weight_; }

private:
    // Reference to the context
    std::shared_ptr<seal::SEALContext> context_;

    // Weights for the transformer
    seal::Plaintext query_weight_;  // W_Q
    seal::Plaintext key_weight_;    // W_K
    seal::Plaintext value_weight_;  // W_V
    seal::Plaintext output_weight_; // W_O
    seal::Plaintext ff1_weight_;    // Feedforward network first layer
    seal::Plaintext ff2_weight_;    // Feedforward network second layer

    // Helper function to load weight matrices
    std::vector<std::vector<double>> loadWeightsFromFile(const std::string& file_path);
};

class EncryptedQuadraticInhibitorAttention {
public:
    EncryptedQuadraticInhibitorAttention(
        std::shared_ptr<seal::SEALContext> context,
        seal::RelinKeys relin_keys,
        seal::GaloisKeys galois_keys,
        int hidden_size,
        int num_attention_heads,
        double scale);
    
    // Forward pass with encrypted inputs and weights
    seal::Ciphertext forward(
        const seal::Ciphertext& input,
        const seal::Ciphertext& wq,
        const seal::Ciphertext& wk,
        const seal::Ciphertext& wv,
        const seal::Ciphertext& wo,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator,
        const seal::Ciphertext* attention_mask = nullptr);
    
    // ReLU activation function for homomorphic encryption
    seal::Ciphertext ReLU(
        const seal::Ciphertext& input,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator);
    
    // Matrix multiply function implementation
    seal::Ciphertext matrixMultiply(
        const seal::Ciphertext& A,
        const seal::Ciphertext& B,
        int rows_A, int cols_A, int cols_B,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator);
    
    // Strassen's matrix multiplication algorithm
    seal::Ciphertext strassenMatrixMultiply(
        const seal::Ciphertext& A,
        const seal::Ciphertext& B,
        int n, // Matrix dimension (assuming square matrices)
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator);
    
    // Helper functions for Strassen's algorithm
    seal::Ciphertext addCiphertext(
        const seal::Ciphertext& A, 
        const seal::Ciphertext& B,
        seal::Evaluator& evaluator);
    
    seal::Ciphertext subtractCiphertext(
        const seal::Ciphertext& A, 
        const seal::Ciphertext& B,
        seal::Evaluator& evaluator);
    
    // Extract submatrix from ciphertext
    seal::Ciphertext extractSubmatrix(
        const seal::Ciphertext& matrix,
        int startRow, int startCol, int size,
        int origSize,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator);
    
    // Combine submatrices into result matrix
    seal::Ciphertext combineSubmatrices(
        const seal::Ciphertext& C11, 
        const seal::Ciphertext& C12,
        const seal::Ciphertext& C21, 
        const seal::Ciphertext& C22,
        int size,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator);
    
    // Approximated ReLU function implementation
    seal::Ciphertext computeApproximatedReLU(
        const seal::Ciphertext& input,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator);
    
private:
    // Compute query-key interactions with quadratic inhibition
    seal::Ciphertext computeQuadraticInhibition(
        const seal::Ciphertext& query,
        const seal::Ciphertext& key,
        const seal::Ciphertext& value,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator,
        const seal::Ciphertext* attention_mask = nullptr);
        
    // Context and keys
    std::shared_ptr<seal::SEALContext> context_;
    seal::RelinKeys relin_keys_;
    seal::GaloisKeys galois_keys_;
    
    // Model parameters
    int hidden_size_;
    int num_attention_heads_;
    int attention_head_size_;
    double scale_; // Scale factor for CKKS encoding
    
    // Gamma coefficient for scaling the quadratic form (will be encrypted)
    double gamma_coef_;
    // Dimension scale term (3d/16)
    double dim_scale_;
};

class EncryptedTransformer {
public:
    EncryptedTransformer(
        std::shared_ptr<seal::SEALContext> context,
        const seal::RelinKeys& relin_keys,
        const seal::GaloisKeys& galois_keys,
        int num_layers,
        int hidden_size,
        int num_attention_heads,
        double scale);
    
    // Initialize with encrypted weights
    void setWeights(std::shared_ptr<EncryptedTransformerWeights> weights);
    
    // Add method to control noise management strategy
    void setRescalingStrategy(bool aggressive_rescaling) {
        aggressive_rescaling_ = aggressive_rescaling;
    }
    
    // Process encrypted input through the transformer
    seal::Ciphertext forward(
        const seal::Ciphertext& input,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator,
        const seal::Ciphertext* attention_mask = nullptr);
    
private:
    // Helper method to rescale ciphertext if needed based on strategy
    void rescaleIfNeeded(
        seal::Ciphertext& cipher,
        seal::Evaluator& evaluator,
        seal::CKKSEncoder& encoder);
        
    // Transformer layer implementation
    seal::Ciphertext transformerLayer(
        const seal::Ciphertext& input,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator,
        seal::Ciphertext* attention_mask);
        
    // Multi-head attention implementation
    seal::Ciphertext multiHeadAttention(
        const seal::Ciphertext& input,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator,
        seal::Ciphertext* attention_mask);
    
    // Layer normalization in encrypted domain (approximation)
    seal::Ciphertext layerNorm(
        const seal::Ciphertext& input,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator);
    
    // Feed forward network in encrypted domain
    seal::Ciphertext feedForward(
        const seal::Ciphertext& input,
        const seal::Ciphertext& ff1_weights,
        const seal::Ciphertext& ff2_weights,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator);
    
    // ReLU activation function approximation
    seal::Ciphertext reluApprox(
        const seal::Ciphertext& input,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator);
        
    // Matrix multiplication for encrypted data
    seal::Ciphertext matrixMultiply(
        const seal::Ciphertext& A,
        const seal::Ciphertext& B,
        int rows_A, int cols_A, int cols_B,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator);
    
    // Components
    std::shared_ptr<seal::SEALContext> context_;
    std::shared_ptr<EncryptedTransformerWeights> weights_;
    std::unique_ptr<EncryptedQuadraticInhibitorAttention> attention_;
    seal::RelinKeys relin_keys_;
    seal::GaloisKeys galois_keys_;
    
    // Parameters
    int num_layers_;
    int hidden_size_;
    int num_attention_heads_;
    double scale_; // Scale factor for CKKS encoding
    
    // Flag to control rescaling strategy
    bool aggressive_rescaling_ = false;
};

class EncryptedInferencePipeline {
public:
    EncryptedInferencePipeline(
        const std::string& model_path,
        int poly_modulus_degree = 65536,
        int num_layers = 12,
        int hidden_size = 768,
        int num_attention_heads = 12);
    
    // Encrypt input, run inference, decrypt output
    std::vector<double> infer(const std::vector<double>& input, const std::vector<bool>& attention_mask = std::vector<bool>());
    
private:
    // Setup SEAL components
    std::shared_ptr<seal::SEALContext> context_;
    seal::SecretKey secret_key_;
    seal::PublicKey public_key_;
    seal::RelinKeys relin_keys_;
    seal::GaloisKeys galois_keys_;
    
    // SEAL operators - using std::unique_ptr to manage objects that don't support assignment
    std::unique_ptr<seal::Encryptor> encryptor_;
    std::unique_ptr<seal::Evaluator> evaluator_;
    std::unique_ptr<seal::Decryptor> decryptor_;
    std::unique_ptr<seal::CKKSEncoder> encoder_;
    
    // Model
    std::shared_ptr<EncryptedTransformerWeights> weights_;
    std::unique_ptr<EncryptedTransformer> transformer_;
    
    // Parameters
    int poly_modulus_degree_;
    int num_layers_;
    int hidden_size_;
    int num_attention_heads_;
    double scale_; // Scale factor for CKKS encoding
}; 