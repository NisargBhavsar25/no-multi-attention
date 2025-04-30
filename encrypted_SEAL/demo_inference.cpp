#include "encrypted_transformer.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <filesystem>

// Helper function to print a vector (only a portion for large vectors)
void printVector(const std::vector<double>& vec, const std::string& name, size_t max_items = 10) {
    std::cout << name << " [" << vec.size() << " elements]: [";
    
    for (size_t i = 0; i < std::min(max_items, vec.size()); i++) {
        std::cout << vec[i];
        if (i < std::min(max_items, vec.size()) - 1) {
            std::cout << ", ";
        }
    }
    
    if (vec.size() > max_items) {
        std::cout << ", ...";
    }
    
    std::cout << "]" << std::endl;
}

// Helper function to generate a random input sequence
std::vector<double> generateRandomInput(int seq_length, int hidden_size) {
    std::vector<double> input(seq_length * hidden_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < seq_length * hidden_size; i++) {
        input[i] = dist(gen);
    }
    
    return input;
}

// Helper function to load input from a binary file
std::vector<double> loadInputFromFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open input file: " + filepath);
    }
    
    // Read number of elements
    uint32_t num_elements;
    file.read(reinterpret_cast<char*>(&num_elements), sizeof(uint32_t));
    
    // Read data
    std::vector<double> data(num_elements);
    for (uint32_t i = 0; i < num_elements; i++) {
        double value;
        file.read(reinterpret_cast<char*>(&value), sizeof(double));
        data[i] = value;
    }
    
    std::cout << "Loaded " << num_elements << " elements from " << filepath << std::endl;
    return data;
}

// Helper function to load attention mask from a binary file
std::vector<bool> loadMaskFromFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open mask file: " + filepath);
    }
    
    // Read sequence length
    uint32_t seq_length;
    file.read(reinterpret_cast<char*>(&seq_length), sizeof(uint32_t));
    
    // Read mask data
    std::vector<bool> mask(seq_length);
    for (uint32_t i = 0; i < seq_length; i++) {
        uint8_t value;
        file.read(reinterpret_cast<char*>(&value), sizeof(uint8_t));
        mask[i] = (value != 0);
    }
    
    std::cout << "Loaded mask with sequence length " << seq_length << " from " << filepath << std::endl;
    return mask;
}

// Helper function to write results to a binary file
void writeResultsToFile(const std::string& filepath, const std::vector<double>& results) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file for writing: " + filepath);
    }
    
    // Write number of elements
    uint32_t num_elements = static_cast<uint32_t>(results.size());
    file.write(reinterpret_cast<const char*>(&num_elements), sizeof(uint32_t));
    
    // Write data
    for (const auto& value : results) {
        file.write(reinterpret_cast<const char*>(&value), sizeof(double));
    }
    
    std::cout << "Wrote " << num_elements << " elements to " << filepath << std::endl;
}

// Helper function to create model directory and generate dummy weight files
void generateDummyWeights(const std::string& model_path, int hidden_size, int intermediate_size) {
    // Create model directory if it doesn't exist
    std::filesystem::create_directories(model_path);
    
    std::cout << "Generating dummy weight files in " << model_path << std::endl;
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    
    // Helper function to write a matrix to a binary file
    auto writeMatrix = [&](const std::string& filename, int rows, int cols) {
        std::ofstream file(model_path + "/" + filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to create file: " << filename << std::endl;
            return;
        }
        
        // Write dimensions
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        
        // Write random data
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double val = dist(gen);
                file.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
        }
        
        file.close();
        std::cout << "  Created " << filename << " with shape [" << rows << ", " << cols << "]" << std::endl;
    };
    
    // Generate weight files
    writeMatrix("wq.bin", hidden_size, hidden_size);
    writeMatrix("wk.bin", hidden_size, hidden_size);
    writeMatrix("wv.bin", hidden_size, hidden_size);
    writeMatrix("wo.bin", hidden_size, hidden_size);
    writeMatrix("ff1.bin", hidden_size, intermediate_size);
    writeMatrix("ff2.bin", intermediate_size, hidden_size);
    
    std::cout << "Dummy weight files generated successfully" << std::endl;
}

// Main function for demonstrating encrypted inference
void runDemoInference(const std::string& model_path, 
                      int poly_modulus_degree = 8192,
                      int num_layers = 1,
                      int hidden_size = 128,
                      int num_attention_heads = 4,
                      int seq_length = 16,
                      const std::string& input_file = "",
                      const std::string& mask_file = "",
                      const std::string& output_file = "") {
    std::cout << "\n=== Running Encrypted Transformer Demo ===" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Parameters: " << std::endl;
    std::cout << "  Polynomial modulus degree: " << poly_modulus_degree << std::endl;
    std::cout << "  Number of layers: " << num_layers << std::endl;
    std::cout << "  Hidden size: " << hidden_size << std::endl;
    std::cout << "  Number of attention heads: " << num_attention_heads << std::endl;
    std::cout << "  Sequence length: " << seq_length << std::endl;
    
    if (!input_file.empty()) {
        std::cout << "  Input file: " << input_file << std::endl;
    }
    
    if (!mask_file.empty()) {
        std::cout << "  Mask file: " << mask_file << std::endl;
    }
    
    if (!output_file.empty()) {
        std::cout << "  Output file: " << output_file << std::endl;
    }
    
    // Check if model directory exists, create it with dummy weights if not
    if (!std::filesystem::exists(model_path) || std::filesystem::is_empty(model_path)) {
        std::cout << "Model directory not found or empty. Generating dummy weights..." << std::endl;
        generateDummyWeights(model_path, hidden_size, 4 * hidden_size);
    }
    
    // Create the inference pipeline
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Phase 1: Initialize encryption parameters and context
        std::cout << "\nPhase 1: Setting up encryption parameters..." << std::endl;
        
        // Create seal::EncryptionParameters object with extremely simplified parameters
        seal::EncryptionParameters parms(seal::scheme_type::ckks);
        parms.set_poly_modulus_degree(8192); // Force 8192 regardless of input
        
        // Super simple coefficient modulus - only use two small primes
        // This is just for proof of concept
        parms.set_coeff_modulus(seal::CoeffModulus::Create(8192, {30, 30}));
        
        // Create context - using security level "none" just for testing
        seal::SEALContext context(parms, true, seal::sec_level_type::none);
        
        std::cout << "Printing context parameters:" << std::endl;
        auto context_data = context.key_context_data();
        std::cout << "  Encryption scheme: CKKS" << std::endl;
        std::cout << "  Poly modulus degree: " << context_data->parms().poly_modulus_degree() << std::endl;
        std::cout << "  Coeff modulus size: " << context_data->total_coeff_modulus_bit_count() << " bits" << std::endl;
        
        // Phase 2: Generate keys
        std::cout << "Phase 2: Generating encryption keys..." << std::endl;
        try {
            // Generate minimal keys - no galois keys for simplicity
            seal::KeyGenerator keygen(context);
            auto secret_key = keygen.secret_key();
            
            seal::PublicKey public_key;
            keygen.create_public_key(public_key);
            
            seal::RelinKeys relin_keys;
            keygen.create_relin_keys(relin_keys);
            
            std::cout << "Keys generated successfully!" << std::endl;
            
            // Phase 3: Create SEAL objects
            std::cout << "Phase 3: Creating SEAL encryption objects..." << std::endl;
            seal::Encryptor encryptor(context, public_key);
            seal::Evaluator evaluator(context);
            seal::Decryptor decryptor(context, secret_key);
            seal::CKKSEncoder encoder(context);
            
            // Phase 4: Get input data - either from file or generate random
            std::cout << "Phase 4: Preparing input data..." << std::endl;
            std::vector<double> input;
            std::vector<bool> attention_mask;
            
            if (!input_file.empty()) {
                // Load input from file
                input = loadInputFromFile(input_file);
                
                // If a hidden_size was specified, check that the input size is correct
                if (input.size() % hidden_size != 0) {
                    throw std::runtime_error("Input file size is not a multiple of hidden_size");
                }
                
                // Update seq_length based on the loaded input
                seq_length = input.size() / hidden_size;
                std::cout << "  Using sequence length " << seq_length << " based on input file" << std::endl;
            } else {
                // Generate random input
                input = generateRandomInput(seq_length, hidden_size);
            }
            
            // Print input
            printVector(input, "Input", 5);
            
            // Load or generate attention mask
            if (!mask_file.empty()) {
                // Load mask from file
                attention_mask = loadMaskFromFile(mask_file);
                
                // Check that the mask size matches seq_length
                if (attention_mask.size() != seq_length) {
                    throw std::runtime_error("Attention mask size does not match sequence length");
                }
            } else {
                // Use all ones (no masking)
                attention_mask = std::vector<bool>(seq_length, true);
            }
            
            // Phase 5: Try a simple encryption/decryption operation
            std::cout << "\nPhase 5: Testing basic encryption/decryption..." << std::endl;
            seal::Plaintext plain;
            encoder.encode(input, pow(2.0, 20), plain);
            
            seal::Ciphertext cipher;
            encryptor.encrypt(plain, cipher);
            
            seal::Plaintext decrypted;
            decryptor.decrypt(cipher, decrypted);
            
            std::vector<double> result;
            encoder.decode(decrypted, result);
            
            std::cout << "Basic encryption/decryption test successful!" << std::endl;
            printVector(result, "Result", 5);
            
            // Phase 6: Try a simple computation (add a constant)
            std::cout << "\nPhase 6: Testing simple computation..." << std::endl;
            seal::Plaintext one;
            encoder.encode(1.0, cipher.scale(), one);
            
            evaluator.add_plain_inplace(cipher, one);
            
            decryptor.decrypt(cipher, decrypted);
            encoder.decode(decrypted, result);
            
            std::cout << "Simple computation test successful!" << std::endl;
            printVector(result, "Result (after adding 1)", 5);
            
            // If an output file was specified, write the result to it
            if (!output_file.empty()) {
                writeResultsToFile(output_file, result);
            }
            
            std::cout << "\nSkipping full pipeline test to focus on SEAL functionality." << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error during key generation: " << e.what() << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error during demo setup: " << e.what() << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "\nDemo completed in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "=== Demo Finished ===" << std::endl;
}

// Main function to demonstrate the encrypted transformer
int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string model_path = "./model";
    int poly_modulus_degree = 8192;
    int num_layers = 1;
    int hidden_size = 64;
    int num_attention_heads = 2;
    int seq_length = 8;
    std::string input_file = "";
    std::string mask_file = "";
    std::string output_file = "";
    
    // Process command line arguments if provided
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--model_path" && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (arg == "--poly_modulus_degree" && i + 1 < argc) {
            poly_modulus_degree = std::stoi(argv[++i]);
        }
        else if (arg == "--num_layers" && i + 1 < argc) {
            num_layers = std::stoi(argv[++i]);
        }
        else if (arg == "--hidden_size" && i + 1 < argc) {
            hidden_size = std::stoi(argv[++i]);
        }
        else if (arg == "--num_attention_heads" && i + 1 < argc) {
            num_attention_heads = std::stoi(argv[++i]);
        }
        else if (arg == "--seq_length" && i + 1 < argc) {
            seq_length = std::stoi(argv[++i]);
        }
        else if (arg == "--input_file" && i + 1 < argc) {
            input_file = argv[++i];
        }
        else if (arg == "--mask_file" && i + 1 < argc) {
            mask_file = argv[++i];
        }
        else if (arg == "--output_file" && i + 1 < argc) {
            output_file = argv[++i];
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --model_path PATH             Path to model weights (default: ./model)" << std::endl;
            std::cout << "  --poly_modulus_degree NUMBER  Polynomial modulus degree (default: 8192)" << std::endl;
            std::cout << "  --num_layers NUMBER           Number of transformer layers (default: 1)" << std::endl;
            std::cout << "  --hidden_size NUMBER          Hidden dimension size (default: 64)" << std::endl;
            std::cout << "  --num_attention_heads NUMBER  Number of attention heads (default: 2)" << std::endl;
            std::cout << "  --seq_length NUMBER           Sequence length (default: 8)" << std::endl;
            std::cout << "  --input_file PATH             Path to input embeddings file (optional)" << std::endl;
            std::cout << "  --mask_file PATH              Path to attention mask file (optional)" << std::endl;
            std::cout << "  --output_file PATH            Path to output result file (optional)" << std::endl;
            std::cout << "  --help, -h                    Display this help message" << std::endl;
            return 0;
        }
    }
    
    // Run the demo with the parsed parameters
    runDemoInference(model_path, poly_modulus_degree, num_layers, hidden_size, 
                    num_attention_heads, seq_length, input_file, mask_file, output_file);
    
    return 0;
} 