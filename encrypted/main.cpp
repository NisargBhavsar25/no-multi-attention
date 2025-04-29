#include "encrypted_transformer.h"
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <chrono>
#include <iomanip>

// Function to generate random test data
std::vector<double> generateRandomData(int size, double min_val = -1.0, double max_val = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min_val, max_val);
    
    std::vector<double> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    
    return data;
}

// Print vector helper
void printVector(const std::vector<double>& vec, const std::string& name, int max_items = 10) {
    std::cout << name << " [" << vec.size() << " elements]: ";
    int to_print = std::min(static_cast<int>(vec.size()), max_items);
    
    for (int i = 0; i < to_print; ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i < to_print - 1) {
            std::cout << ", ";
        }
    }
    
    if (vec.size() > max_items) {
        std::cout << ", ...";
    }
    
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        std::string model_path = "./model";
        int poly_modulus_degree = 65536;
        int num_layers = 1;  // Using a single layer for testing
        int hidden_size = 128;  // Smaller size for testing
        int num_attention_heads = 4;
        
        // Process command line args if provided
        for (int i = 1; i < argc; i += 2) {
            std::string arg = argv[i];
            if (i + 1 < argc) {
                if (arg == "--model_path") {
                    model_path = argv[i + 1];
                } else if (arg == "--poly_modulus_degree") {
                    poly_modulus_degree = std::stoi(argv[i + 1]);
                } else if (arg == "--num_layers") {
                    num_layers = std::stoi(argv[i + 1]);
                } else if (arg == "--hidden_size") {
                    hidden_size = std::stoi(argv[i + 1]);
                } else if (arg == "--num_attention_heads") {
                    num_attention_heads = std::stoi(argv[i + 1]);
                }
            }
        }
        
        std::cout << "========================================================" << std::endl;
        std::cout << "   Quadratic Inhibitor Transformer Encryption Demo" << std::endl;
        std::cout << "========================================================" << std::endl;
        std::cout << "Model path: " << model_path << std::endl;
        std::cout << "Polynomial modulus degree: " << poly_modulus_degree << std::endl;
        std::cout << "Number of layers: " << num_layers << std::endl;
        std::cout << "Hidden size: " << hidden_size << std::endl;
        std::cout << "Number of attention heads: " << num_attention_heads << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
        
        // Create and initialize the encrypted inference pipeline
        auto start_time = std::chrono::high_resolution_clock::now();
        
        EncryptedInferencePipeline pipeline(
            model_path,
            poly_modulus_degree,
            num_layers,
            hidden_size,
            num_attention_heads);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        std::cout << "Pipeline setup completed in " << setup_duration << " ms" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
        
        // Generate random input data (for demo purposes)
        std::cout << "Generating random input data..." << std::endl;
        std::vector<double> input_data = generateRandomData(hidden_size, -0.5, 0.5);
        printVector(input_data, "Input", 5);
        
        // Run inference
        std::cout << "Running encrypted inference..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<double> output_data = pipeline.infer(input_data);
        
        end_time = std::chrono::high_resolution_clock::now();
        auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        std::cout << "Inference completed in " << inference_duration << " ms" << std::endl;
        printVector(output_data, "Output", 5);
        
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "Demo completed successfully!" << std::endl;
        std::cout << "Total time: " << (setup_duration + inference_duration) << " ms" << std::endl;
        std::cout << "========================================================" << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 