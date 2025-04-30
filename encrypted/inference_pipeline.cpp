#include "encrypted_transformer.h"
#include <iostream>
#include <chrono>

EncryptedInferencePipeline::EncryptedInferencePipeline(
    const std::string& model_path,
    int poly_modulus_degree,
    int num_layers,
    int hidden_size,
    int num_attention_heads)
    : poly_modulus_degree_(poly_modulus_degree),
      num_layers_(num_layers),
      hidden_size_(hidden_size),
      num_attention_heads_(num_attention_heads) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Initializing encrypted inference pipeline..." << std::endl;
    
    // Initialize CKKS context
    std::cout << "Setting up CKKS context with poly_modulus_degree = " << poly_modulus_degree_ << std::endl;
    context_.set_poly_modulus_degree(poly_modulus_degree_);
    
    // Set up coefficient modulus
    // For deep networks, we need a large number of levels
    // The actual number depends on the depth of computations in the network
    // For a typical transformer, at least 5 levels are needed
    context_.set_coeff_modulus_values(5);
    
    // Generate context
    context_.generate();
    
    // Generate keys
    std::cout << "Generating encryption keys..." << std::endl;
    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context_);
    
    // Secret key
    keygen.generate_secret_key(secret_key_);
    
    // Public key
    keygen.generate_public_key(public_key_, secret_key_);
    
    // Relinearization key (needed for multiplication operations)
    keygen.generate_relin_key(relin_key_, secret_key_);
    
    // Galois keys (needed for advanced operations like rotations)
    keygen.generate_galois_key(galois_key_, secret_key_);
    
    // Initialize encoder, encryptor, decryptor
    encoder_ = heongpu::HEEncoder<heongpu::Scheme::CKKS>(context_);
    encryptor_ = heongpu::HEEncryptor<heongpu::Scheme::CKKS>(context_, public_key_);
    decryptor_ = heongpu::HEDecryptor<heongpu::Scheme::CKKS>(context_, secret_key_);
    
    // Set scale for CKKS encoding (affects precision)
    scale_ = pow(2.0, 40);
    
    // Load and encrypt model weights
    std::cout << "Loading and encrypting model weights..." << std::endl;
    weights_ = std::make_shared<EncryptedTransformerWeights>(context_);
    weights_->loadFromPretrained(model_path, encryptor_, encoder_, scale_);
    
    // Initialize transformer
    std::cout << "Initializing transformer model..." << std::endl;
    transformer_ = std::make_unique<EncryptedTransformer>(
        context_,
        relin_key_,
        galois_key_,
        num_layers_,
        hidden_size_,
        num_attention_heads_,
        scale_);
    
    // Set weights for transformer
    transformer_->setWeights(weights_);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "Encrypted inference pipeline initialized in " << duration << " ms" << std::endl;
}

std::vector<double> EncryptedInferencePipeline::infer(const std::vector<double>& input, const std::vector<bool>& attention_mask) {
    if (input.size() != hidden_size_) {
        throw std::runtime_error("Input size mismatch: expected " + std::to_string(hidden_size_) + 
                                 ", got " + std::to_string(input.size()));
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Starting encrypted inference..." << std::endl;
    
    // 1. Encode input
    std::cout << "Encoding input data..." << std::endl;
    heongpu::Plaintext<heongpu::Scheme::CKKS> input_plain(context_);
    encoder_.encode(input_plain, input, scale_);
    
    // 2. Encrypt input
    std::cout << "Encrypting input data..." << std::endl;
    heongpu::Ciphertext<heongpu::Scheme::CKKS> input_cipher(context_);
    encryptor_.encrypt(input_cipher, input_plain);
    
    // 3. Run the transformer model
    std::cout << "Processing through transformer model..." << std::endl;
    heongpu::Ciphertext<heongpu::Scheme::CKKS> output_cipher = 
        transformer_->forward(input_cipher, encoder_, encryptor_);
    
    // 4. Decrypt output
    std::cout << "Decrypting output..." << std::endl;
    heongpu::Plaintext<heongpu::Scheme::CKKS> output_plain(context_);
    decryptor_.decrypt(output_plain, output_cipher);
    
    // 5. Decode output
    std::cout << "Decoding output data..." << std::endl;
    std::vector<double> output;
    encoder_.decode(output, output_plain);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "Encrypted inference completed in " << duration << " ms" << std::endl;
    
    return output;
} 