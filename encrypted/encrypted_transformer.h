#pragma once

#include <heongpu.cuh>
#include <vector>
#include <string>
#include <memory>

/**
 * Encrypted Quadratic Inhibitor Transformer Implementation
 * 
 * This implements a transformer with quadratic inhibitor attention mechanism
 * using the HEonGPU library for homomorphic encryption operations.
 */

class EncryptedTransformerWeights {
public:
    EncryptedTransformerWeights(heongpu::HEContext<heongpu::Scheme::CKKS>& context);

    // Load and encrypt weights from pretrained model
    void loadFromPretrained(
        const std::string& model_path,
        heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor,
        heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
        double scale);

    // Get weights for different transformer components
    const std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>>& getQueryWeights() const { return wq_weights; }
    const std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>>& getKeyWeights() const { return wk_weights; }
    const std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>>& getValueWeights() const { return wv_weights; }
    const std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>>& getOutputWeights() const { return wo_weights; }
    const std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>>& getFF1Weights() const { return ff1_weights; }
    const std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>>& getFF2Weights() const { return ff2_weights; }

private:
    // Reference to the context
    heongpu::HEContext<heongpu::Scheme::CKKS>& context_;

    // Query, Key, Value weights for each layer
    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> wq_weights;
    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> wk_weights;
    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> wv_weights;
    
    // Output projection weights
    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> wo_weights;
    
    // Feed-forward weights
    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> ff1_weights;
    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> ff2_weights;

    // Helper function to load weight matrices
    std::vector<std::vector<double>> loadWeightsFromFile(const std::string& file_path);
};

class EncryptedQuadraticInhibitorAttention {
public:
    EncryptedQuadraticInhibitorAttention(
        heongpu::HEContext<heongpu::Scheme::CKKS>& context,
        heongpu::Relinkey<heongpu::Scheme::CKKS>& relin_key,
        heongpu::Galoiskey<heongpu::Scheme::CKKS>& galois_key,
        int hidden_size,
        int num_attention_heads,
        double scale);
    
    // Forward pass with encrypted inputs and weights
    heongpu::Ciphertext<heongpu::Scheme::CKKS> forward(
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& wq,
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& wk,
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& wv,
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& wo,
        heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
        heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor,
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>* attention_mask = nullptr);
    
    // ReLU activation function for homomorphic encryption
    heongpu::Ciphertext<heongpu::Scheme::CKKS> ReLU(
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
        heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
        heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor);
    
    // Matrix multiply function implementation
    heongpu::Ciphertext<heongpu::Scheme::CKKS> matrixMultiply(
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& A,
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& B,
        int rows_A, int cols_A, int cols_B,
        heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
        heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor);
    
    // Approximated ReLU function implementation
    heongpu::Ciphertext<heongpu::Scheme::CKKS> computeApproximatedReLU(
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
        heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
        heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor);
    
private:
    // Compute query-key interactions with quadratic inhibition
    heongpu::Ciphertext<heongpu::Scheme::CKKS> computeQuadraticInhibition(
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& query,
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& key,
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& value,
        heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
        heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor,
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>* attention_mask = nullptr);
        
    // Context and operators
    heongpu::HEContext<heongpu::Scheme::CKKS>& context_;
    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder_;
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators_;
    heongpu::Relinkey<heongpu::Scheme::CKKS>& relin_key_;
    heongpu::Galoiskey<heongpu::Scheme::CKKS>& galois_key_;
    
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
        heongpu::HEContext<heongpu::Scheme::CKKS>& context,
        heongpu::Relinkey<heongpu::Scheme::CKKS>& relin_key,
        heongpu::Galoiskey<heongpu::Scheme::CKKS>& galois_key,
        int num_layers,
        int hidden_size,
        int num_attention_heads,
        double scale);
    
    // Initialize with encrypted weights
    void setWeights(std::shared_ptr<EncryptedTransformerWeights> weights);
    
    // Process encrypted input through the transformer
    heongpu::Ciphertext<heongpu::Scheme::CKKS> forward(
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
        heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
        heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor,
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>* attention_mask = nullptr);
    
private:
    // Layer normalization in encrypted domain (approximation)
    heongpu::Ciphertext<heongpu::Scheme::CKKS> layerNorm(
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
        heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
        heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor);
    
    // Feed forward network in encrypted domain
    heongpu::Ciphertext<heongpu::Scheme::CKKS> feedForward(
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& ff1_weights,
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& ff2_weights,
        heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
        heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor);
    
    // ReLU activation function approximation
    heongpu::Ciphertext<heongpu::Scheme::CKKS> reluApprox(
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& input,
        heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
        heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor);
        
    // Matrix multiplication for encrypted data
    // Computes A * B where A is encrypted and B is encrypted
    heongpu::Ciphertext<heongpu::Scheme::CKKS> matrixMultiply(
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& A,
        const heongpu::Ciphertext<heongpu::Scheme::CKKS>& B,
        int rows_A, int cols_A, int cols_B,
        heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder,
        heongpu::HEEncryptor<heongpu::Scheme::CKKS>& encryptor);
    
    // Components
    heongpu::HEContext<heongpu::Scheme::CKKS>& context_;
    std::shared_ptr<EncryptedTransformerWeights> weights_;
    std::unique_ptr<EncryptedQuadraticInhibitorAttention> attention_;
    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder_;
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators_;
    heongpu::Relinkey<heongpu::Scheme::CKKS>& relin_key_;
    heongpu::Galoiskey<heongpu::Scheme::CKKS>& galois_key_;
    
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
    // Setup
    heongpu::HEContext<heongpu::Scheme::CKKS> context_;
    heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key_;
    heongpu::Publickey<heongpu::Scheme::CKKS> public_key_;
    heongpu::Relinkey<heongpu::Scheme::CKKS> relin_key_;
    heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key_;
    
    // Operators
    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder_;
    heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor_;
    heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor_;
    
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