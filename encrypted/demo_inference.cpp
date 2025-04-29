#include "encrypted_transformer.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

int main(int argc, char** argv) {
    // Default parameters
    std::string model_path = "../converted_model_quadratic_relu";
    int poly_modulus_degree = 65536;
    int num_layers = 4;
    int hidden_size = 768;
    int num_attention_heads = 1;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model_path" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--poly_modulus_degree" && i + 1 < argc) {
            poly_modulus_degree = std::stoi(argv[++i]);
        } else if (arg == "--num_layers" && i + 1 < argc) {
            num_layers = std::stoi(argv[++i]);
        } else if (arg == "--hidden_size" && i + 1 < argc) {
            hidden_size = std::stoi(argv[++i]);
        } else if (arg == "--num_attention_heads" && i + 1 < argc) {
            num_attention_heads = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --model_path PATH             Path to the converted model (default: ../converted_model_quadratic_relu)\n"
                      << "  --poly_modulus_degree NUMBER  Polynomial modulus degree (default: 65536)\n"
                      << "  --num_layers NUMBER           Number of transformer layers (default: 4)\n"
                      << "  --hidden_size NUMBER          Hidden size (default: 768)\n"
                      << "  --num_attention_heads NUMBER  Number of attention heads (default: 1)\n";
            return 0;
        }
    }

    std::cout << "Initializing Encrypted Inference Pipeline...\n";
    std::cout << "Model path: " << model_path << "\n";
    std::cout << "Polynomial modulus degree: " << poly_modulus_degree << "\n";
    std::cout << "Number of layers: " << num_layers << "\n";
    std::cout << "Hidden size: " << hidden_size << "\n";
    std::cout << "Number of attention heads: " << num_attention_heads << "\n";

    try {
        // Initialize the encrypted inference pipeline
        EncryptedInferencePipeline pipeline(
            model_path,
            poly_modulus_degree,
            num_layers,
            hidden_size,
            num_attention_heads
        );

        // Create a mock input vector (size = hidden_size)
        // In a real scenario, this would be your input embedding
        std::vector<float> input(hidden_size);
        for (int i = 0; i < hidden_size; i++) {
            input[i] = 0.01f * (i % 10); // Simple pattern for testing
        }

        std::cout << "Input preview (first 10 values):\n";
        for (int i = 0; i < std::min(10, hidden_size); i++) {
            std::cout << input[i] << " ";
        }
        std::cout << "\n";

        // Create attention mask (all true for simplicity)
        std::vector<bool> attention_mask(hidden_size, true);

        // Run inference and measure time
        std::cout << "Running inference...\n";
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<float> output = pipeline.infer(input, attention_mask);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Print results
        std::cout << "Inference completed in " << duration.count() << " ms\n";
        std::cout << "Output size: " << output.size() << "\n";
        std::cout << "Output preview (first 10 values):\n";
        for (int i = 0; i < std::min(10, static_cast<int>(output.size())); i++) {
            std::cout << output[i] << " ";
        }
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 