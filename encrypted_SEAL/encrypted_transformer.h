#pragma once

#include <seal/seal.h>
#include <vector>
#include <string>
#include <memory>

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

    // Get weights for different transformer components
    const std::vector<seal::Ciphertext>& getQueryWeights() const { return wq_weights; }
    const std::vector<seal::Ciphertext>& getKeyWeights() const { return wk_weights; }
    const std::vector<seal::Ciphertext>& getValueWeights() const { return wv_weights; }
    const std::vector<seal::Ciphertext>& getOutputWeights() const { return wo_weights; }
    const std::vector<seal::Ciphertext>& getFF1Weights() const { return ff1_weights; }
    const std::vector<seal::Ciphertext>& getFF2Weights() const { return ff2_weights; }

private:
    // Reference to the context
    std::shared_ptr<seal::SEALContext> context_;

    // Query, Key, Value weights for each layer
    std::vector<seal::Ciphertext> wq_weights;
    std::vector<seal::Ciphertext> wk_weights;
    std::vector<seal::Ciphertext> wv_weights;
    
    // Output projection weights
    std::vector<seal::Ciphertext> wo_weights;
    
    // Feed-forward weights
    std::vector<seal::Ciphertext> ff1_weights;
    std::vector<seal::Ciphertext> ff2_weights;

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
        seal::RelinKeys relin_keys,
        seal::GaloisKeys galois_keys,
        int num_layers,
        int hidden_size,
        int num_attention_heads,
        double scale);
    
    // Initialize with encrypted weights
    void setWeights(std::shared_ptr<EncryptedTransformerWeights> weights);
    
    // Process encrypted input through the transformer
    seal::Ciphertext forward(
        const seal::Ciphertext& input,
        seal::CKKSEncoder& encoder,
        seal::Encryptor& encryptor,
        seal::Evaluator& evaluator,
        const seal::Ciphertext* attention_mask = nullptr);
    
private:
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