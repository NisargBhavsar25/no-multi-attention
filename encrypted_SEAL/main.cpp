#include "encrypted_transformer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <functional>
#include <memory>
#include <stdexcept>

// Command line argument parsing helper
std::string getCmdOption(const std::vector<std::string>& args, const std::string& option, const std::string& defaultValue) {
    auto it = std::find(args.begin(), args.end(), option);
    if (it != args.end() && ++it != args.end()) {
        return *it;
    }
    return defaultValue;
}

// Check if command line option exists
bool cmdOptionExists(const std::vector<std::string>& args, const std::string& option) {
    return std::find(args.begin(), args.end(), option) != args.end();
}

// Generate random vector for testing
std::vector<double> generateRandomVector(size_t size, double min = -1.0, double max = 1.0) {
    std::vector<double> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);
    
    for (size_t i = 0; i < size; i++) {
        vec[i] = dist(gen);
    }
    
    return vec;
}

// Print helper for vector
void printVector(const std::vector<double>& vec, size_t max_display = 10) {
    std::cout << "[";
    
    size_t display_count = std::min(max_display, vec.size());
    
    for (size_t i = 0; i < display_count; i++) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i < display_count - 1) {
            std::cout << ", ";
        }
    }
    
    if (vec.size() > max_display) {
        std::cout << ", ... (" << vec.size() - max_display << " more elements)";
    }
    
    std::cout << "]" << std::endl;
}

// Main function
int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        std::vector<std::string> args(argv, argv + argc);
        
        std::string model_path = getCmdOption(args, "--model_path", "./model");
        int poly_modulus_degree = std::stoi(getCmdOption(args, "--poly_modulus_degree", "65536"));
        int num_layers = std::stoi(getCmdOption(args, "--num_layers", "1"));
        int hidden_size = std::stoi(getCmdOption(args, "--hidden_size", "128"));
        int num_attention_heads = std::stoi(getCmdOption(args, "--num_attention_heads", "4"));
        int seq_length = std::stoi(getCmdOption(args, "--seq_length", "16"));
        bool verbose = cmdOptionExists(args, "--verbose");
        bool help = cmdOptionExists(args, "--help") || cmdOptionExists(args, "-h");
        
        if (help) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --model_path PATH             Path to model weights (default: ./model)" << std::endl;
            std::cout << "  --poly_modulus_degree NUMBER  Polynomial modulus degree (default: 65536)" << std::endl;
            std::cout << "  --num_layers NUMBER           Number of transformer layers (default: 1)" << std::endl;
            std::cout << "  --hidden_size NUMBER          Hidden dimension size (default: 128)" << std::endl;
            std::cout << "  --num_attention_heads NUMBER  Number of attention heads (default: 4)" << std::endl;
            std::cout << "  --seq_length NUMBER           Input sequence length (default: 16)" << std::endl;
            std::cout << "  --verbose                     Enable verbose output" << std::endl;
            std::cout << "  --help, -h                    Display this help message" << std::endl;
            return 0;
        }
        
        // Print parameters
        std::cout << "=== Encrypted Transformer Parameters ===" << std::endl;
        std::cout << "Model Path: " << model_path << std::endl;
        std::cout << "Polynomial Modulus Degree: " << poly_modulus_degree << std::endl;
        std::cout << "Number of Layers: " << num_layers << std::endl;
        std::cout << "Hidden Size: " << hidden_size << std::endl;
        std::cout << "Number of Attention Heads: " << num_attention_heads << std::endl;
        std::cout << "Sequence Length: " << seq_length << std::endl;
        std::cout << "=======================================" << std::endl;
        
        // Create the inference pipeline
        auto start_setup = std::chrono::high_resolution_clock::now();
        
        std::cout << "Initializing encrypted inference pipeline..." << std::endl;
        EncryptedInferencePipeline pipeline(
            model_path,
            poly_modulus_degree,
            num_layers,
            hidden_size,
            num_attention_heads
        );
        
        auto end_setup = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> setup_time = end_setup - start_setup;
        std::cout << "Pipeline setup time: " << setup_time.count() << " seconds" << std::endl;
        
        // Generate a random input vector
        std::cout << "Generating random input..." << std::endl;
        std::vector<double> input = generateRandomVector(seq_length * hidden_size);
        
        // Create an attention mask (all positions visible for this example)
        std::vector<bool> attention_mask(seq_length, true);
        
        if (verbose) {
            std::cout << "Input vector (sample): ";
            printVector(input, 10);
        }
        
        // Run inference
        std::cout << "Running encrypted inference..." << std::endl;
        auto start_inference = std::chrono::high_resolution_clock::now();
        
        std::vector<double> output = pipeline.infer(input, attention_mask);
        
        auto end_inference = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> inference_time = end_inference - start_inference;
        
        // Print results
        std::cout << "Inference completed in " << inference_time.count() << " seconds" << std::endl;
        std::cout << "Output vector (sample): ";
        printVector(output, 10);
        
        // Calculate and print statistics
        double total_time = setup_time.count() + inference_time.count();
        std::cout << "=== Performance Summary ===" << std::endl;
        std::cout << "Total execution time: " << total_time << " seconds" << std::endl;
        std::cout << "  - Setup time: " << setup_time.count() << " seconds (" 
                  << (setup_time.count() / total_time * 100.0) << "%)" << std::endl;
        std::cout << "  - Inference time: " << inference_time.count() << " seconds (" 
                  << (inference_time.count() / total_time * 100.0) << "%)" << std::endl;
        std::cout << "=========================" << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 