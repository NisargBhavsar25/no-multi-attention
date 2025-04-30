#include "encrypted_transformer.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>

// Constructor for the encrypted inference pipeline
EncryptedInferencePipeline::EncryptedInferencePipeline(
    const std::string& model_path,
    int poly_modulus_degree,
    int num_layers,
    int hidden_size,
    int num_attention_heads)
    : poly_modulus_degree_(poly_modulus_degree),
      num_layers_(num_layers),
      hidden_size_(hidden_size),
      num_attention_heads_(num_attention_heads),
      scale_(pow(2.0, 30)) {  // Reduced from 2^40 to 2^30 for better stability
    
    std::cout << "Initializing EncryptedInferencePipeline with:"
              << "\n  Model Path: " << model_path
              << "\n  Poly Modulus Degree: " << poly_modulus_degree_
              << "\n  Num Layers: " << num_layers_
              << "\n  Hidden Size: " << hidden_size_
              << "\n  Num Attention Heads: " << num_attention_heads_
              << std::endl;
    
    // Set up the SEAL context with CKKS scheme
    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree_);
    
    // Set the coefficient modulus for CKKS - use simpler coefficient modulus chain
    // For CKKS, we need enough prime factors for multiply depth
    // Use fewer primes with smaller bit-size for better performance and security
    if (poly_modulus_degree_ == 8192) {
        parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree_, {40, 30, 30, 40}));
    } else if (poly_modulus_degree_ == 16384) {
        parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree_, {40, 30, 30, 30, 40}));
    } else if (poly_modulus_degree_ == 32768) {
        parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree_, {40, 30, 30, 30, 30, 40}));
    } else if (poly_modulus_degree_ == 65536) {
        parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree_, {40, 30, 30, 30, 30, 30, 40}));
    } else {
        throw std::invalid_argument("Unsupported polynomial modulus degree");
    }
    
    // Create the SEAL context - use a lower security level for testing
    context_ = std::make_shared<seal::SEALContext>(parms, true, seal::sec_level_type::none);
    
    // Print key generation info
    std::cout << "Generating keys..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create the key generator
    seal::KeyGenerator keygen(*context_);
    
    // Get the secret key
    secret_key_ = keygen.secret_key();
    
    // Generate the public key
    keygen.create_public_key(public_key_);
    
    // Generate relinearization keys
    keygen.create_relin_keys(relin_keys_);
    
    // Generate Galois keys for rotation operations - use fewer elements
    std::vector<int> galois_elts;
    for (int i = 0; i < 4; i++) {  // Reduced from 8 to 4
        galois_elts.push_back((poly_modulus_degree_ / 2) + i);
    }
    keygen.create_galois_keys(galois_elts, galois_keys_);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Key generation completed in " << elapsed.count() << " seconds" << std::endl;
    
    // Create SEAL objects using unique_ptr
    encryptor_ = std::make_unique<seal::Encryptor>(*context_, public_key_);
    evaluator_ = std::make_unique<seal::Evaluator>(*context_);
    decryptor_ = std::make_unique<seal::Decryptor>(*context_, secret_key_);
    encoder_ = std::make_unique<seal::CKKSEncoder>(*context_);
    
    // Create and set up the transformer components
    weights_ = std::make_shared<EncryptedTransformerWeights>(context_);
    weights_->loadFromPretrained(model_path, *encryptor_, *encoder_, scale_);
    
    transformer_ = std::make_unique<EncryptedTransformer>(
        context_, relin_keys_, galois_keys_, num_layers_, hidden_size_, num_attention_heads_, scale_);
    transformer_->setWeights(weights_);
    
    std::cout << "Encrypted inference pipeline initialized successfully" << std::endl;
}

// Main inference method
std::vector<double> EncryptedInferencePipeline::infer(
    const std::vector<double>& input,
    const std::vector<bool>& attention_mask) {
    
    if (input.size() % hidden_size_ != 0) {
        throw std::invalid_argument("Input size must be a multiple of hidden_size");
    }
    
    // Calculate the sequence length
    const size_t seq_length = input.size() / hidden_size_;
    std::cout << "Running inference on input of sequence length: " << seq_length << std::endl;
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Step 1: Encode and encrypt the input
    std::cout << "Encoding and encrypting input..." << std::endl;
    seal::Plaintext input_plain;
    encoder_->encode(input, scale_, input_plain);
    
    seal::Ciphertext input_cipher;
    encryptor_->encrypt(input_plain, input_cipher);
    
    // Step 2: Prepare the attention mask if provided
    seal::Ciphertext* attention_mask_cipher = nullptr;
    std::unique_ptr<seal::Ciphertext> attention_mask_ptr;
    
    if (!attention_mask.empty()) {
        if (attention_mask.size() != seq_length) {
            throw std::invalid_argument("Attention mask size must match sequence length");
        }
        
        // Convert bool mask to doubles (1.0 for visible, 0.0 for masked positions)
        std::vector<double> mask_values(seq_length * seq_length, 0.0);
        for (size_t i = 0; i < seq_length; i++) {
            for (size_t j = 0; j < seq_length; j++) {
                if (attention_mask[i] && attention_mask[j]) {
                    mask_values[i * seq_length + j] = 1.0;
                }
            }
        }
        
        // Encode and encrypt the mask
        seal::Plaintext mask_plain;
        encoder_->encode(mask_values, scale_, mask_plain);
        
        attention_mask_ptr = std::make_unique<seal::Ciphertext>();
        encryptor_->encrypt(mask_plain, *attention_mask_ptr);
        attention_mask_cipher = attention_mask_ptr.get();
    }
    
    // Step 3: Pass through the transformer
    std::cout << "Running transformer forward pass..." << std::endl;
    seal::Ciphertext output_cipher = transformer_->forward(
        input_cipher, *encoder_, *encryptor_, *evaluator_, attention_mask_cipher);
    
    // Step 4: Decrypt the output
    std::cout << "Decrypting output..." << std::endl;
    seal::Plaintext output_plain;
    decryptor_->decrypt(output_cipher, output_plain);
    
    // Decode the output
    std::vector<double> output;
    encoder_->decode(output_plain, output);
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Inference completed in " << elapsed.count() << " seconds" << std::endl;
    return output;
} 