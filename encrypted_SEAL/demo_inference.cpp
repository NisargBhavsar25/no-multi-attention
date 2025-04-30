#include "encrypted_transformer.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>

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

// Main function to run the actual inference pipeline
void runInferencePipeline(const std::string& model_path, 
                         int poly_modulus_degree,
                         int num_layers,
                         int hidden_size,
                         int num_attention_heads,
                         int seq_length,
                         const std::string& input_file,
                         const std::string& mask_file,
                         const std::string& output_file) {
    
    std::cout << "\n=== Running Full Encrypted Transformer Pipeline ===" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Parameters: " << std::endl;
    std::cout << "  Polynomial modulus degree: " << poly_modulus_degree << std::endl;
    std::cout << "  Number of layers: " << num_layers << std::endl;
    std::cout << "  Hidden size: " << hidden_size << std::endl;
    std::cout << "  Number of attention heads: " << num_attention_heads << std::endl;
    std::cout << "  Sequence length: " << seq_length << std::endl;
    
    // Check if Strassen's algorithm will be used
    bool can_use_strassen = (hidden_size & (hidden_size - 1)) == 0; // Check if power of 2
    std::cout << "  Matrix multiplication: " 
              << (can_use_strassen ? "Strassen's algorithm (O(n^2.807))" : "Standard multiplication (O(n^3))") 
              << std::endl;
    
    if (input_file.empty() || mask_file.empty() || output_file.empty()) {
        throw std::invalid_argument("Input file, mask file, and output file must all be provided");
    }
    
    std::cout << "  Input file: " << input_file << std::endl;
    std::cout << "  Mask file: " << mask_file << std::endl;
    std::cout << "  Output file: " << output_file << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Load input data
        std::cout << "Loading input embeddings..." << std::endl;
        std::vector<double> input = loadInputFromFile(input_file);
        
        // Check if sequence length is correct
        if (input.size() % hidden_size != 0) {
            throw std::runtime_error("Input file size is not a multiple of hidden_size");
        }
        
        int actual_seq_length = input.size() / hidden_size;
        std::cout << "Actual sequence length from input: " << actual_seq_length << std::endl;
        
        // Load attention mask
        std::cout << "Loading attention mask..." << std::endl;
        std::vector<bool> attention_mask = loadMaskFromFile(mask_file);
        
        // Check that attention mask length matches sequence length
        if (attention_mask.size() != actual_seq_length) {
            std::cout << "Warning: Attention mask size (" << attention_mask.size() 
                      << ") does not match sequence length (" << actual_seq_length << ")" << std::endl;
            
            // Adjust the attention mask if needed
            if (attention_mask.size() < actual_seq_length) {
                // Extend with true values (no masking)
                attention_mask.resize(actual_seq_length, true);
                std::cout << "Extended attention mask to match sequence length" << std::endl;
            } else {
                // Truncate
                attention_mask.resize(actual_seq_length);
                std::cout << "Truncated attention mask to match sequence length" << std::endl;
            }
        }
        
        // Create and initialize the pipeline
        std::cout << "Initializing encryption pipeline..." << std::endl;
        EncryptedInferencePipeline pipeline(
            model_path,
            poly_modulus_degree,
            num_layers,
            hidden_size,
            num_attention_heads
        );
        
        // Run inference
        std::cout << "Running encrypted inference..." << std::endl;
        std::vector<double> output = pipeline.infer(input, attention_mask);
        
        // Write output to file
        std::cout << "Writing results to output file..." << std::endl;
        writeResultsToFile(output_file, output);
        
        // Print sample of output
        printVector(output, "Output", 10);
    }
    catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        throw; // Re-throw to maintain the exception
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "\nInference completed in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "=== Pipeline Execution Finished ===" << std::endl;
}

// Main function
int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string model_path = "./model";
    int poly_modulus_degree = 131072; // Using larger degree as requested
    int num_layers = 1;
    int hidden_size = 64;
    int num_attention_heads = 2;
    int seq_length = 16;
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
            std::cout << "  --poly_modulus_degree NUMBER  Polynomial modulus degree (default: 131072)" << std::endl;
            std::cout << "  --num_layers NUMBER           Number of transformer layers (default: 1)" << std::endl;
            std::cout << "  --hidden_size NUMBER          Hidden dimension size (default: 64)" << std::endl;
            std::cout << "  --num_attention_heads NUMBER  Number of attention heads (default: 2)" << std::endl;
            std::cout << "  --seq_length NUMBER           Sequence length (default: 16)" << std::endl;
            std::cout << "  --input_file PATH             Path to input embeddings file (required)" << std::endl;
            std::cout << "  --mask_file PATH              Path to attention mask file (required)" << std::endl;
            std::cout << "  --output_file PATH            Path to output result file (required)" << std::endl;
            std::cout << "  --help, -h                    Display this help message" << std::endl;
            return 0;
        }
    }
    
    // Run the inference pipeline with the parsed parameters
    try {
        runInferencePipeline(
            model_path, 
            poly_modulus_degree, 
            num_layers, 
            hidden_size, 
            num_attention_heads, 
            seq_length, 
            input_file, 
            mask_file, 
            output_file
        );
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 