# Encrypted Quadratic Inhibitor Transformer

This project implements a Fully Homomorphic Encryption (FHE) version of a transformer model with quadratic inhibitor attention mechanism using the HEonGPU library. It enables running transformer models on encrypted data, preserving privacy throughout the inference process.

## Overview

The implementation provides an encrypted version of the quadratic inhibitor attention mechanism, which uses a quadratic form approximation of inhibition that's compatible with homomorphic encryption operations. This approach is well-suited for the CKKS encryption scheme since it requires only polynomial operations.

The encrypted transformer consists of the following components:

1. **Encrypted Transformer Weights**: Handles loading and encrypting pretrained model weights
2. **Encrypted Quadratic Inhibitor Attention**: Implements the quadratic inhibitor attention mechanism in the encrypted domain
3. **Encrypted Transformer**: Manages the complete transformer model with multiple layers
4. **Encrypted Inference Pipeline**: Provides an end-to-end solution for encrypted inference

## Prerequisites

- [CMake](https://cmake.org/download/) >=3.26
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) >=11.4
- [HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU)
- Compatible NVIDIA GPU (Tested on RTX series)

## Building the Project

1. Make sure HEonGPU is properly installed on your system
2. Clone this repository
3. Build the project:

```bash
mkdir build
cd build
cmake ..
make
```

This will generate the executable `encrypted_transformer` in the `build` directory.

## Testing with Dummy Weights

For testing purposes, you can generate dummy weight files:

```bash
cd build
make generate_test_weights
```

## Running the Demo

```bash
./encrypted_transformer --model_path ./model --poly_modulus_degree 65536 --num_layers 1 --hidden_size 128 --num_attention_heads 4
```

Command line options:
- `--model_path`: Path to the pretrained weights (default: ./model)
- `--poly_modulus_degree`: CKKS polynomial modulus degree (default: 65536)
- `--num_layers`: Number of transformer layers (default: 1)
- `--hidden_size`: Hidden dimension size (default: 128)
- `--num_attention_heads`: Number of attention heads (default: 4)

## Loading Real Weights

To use real weights from a trained model:

1. Convert the PyTorch weights to the binary format expected by this implementation
2. Place the weight files in the model directory with the following names:
   - `wq.bin`: Query projection weights
   - `wk.bin`: Key projection weights
   - `wv.bin`: Value projection weights
   - `wo.bin`: Output projection weights
   - `ff1.bin`: Feed-forward first layer weights
   - `ff2.bin`: Feed-forward second layer weights

## Implementation Details

### Quadratic Inhibitor Attention

The quadratic inhibitor attention is implemented as:

```
H_ik ≈ ∑_j (V_jk - (15/(16γ) ||Q_i - K_j||²_2 + 3d/(16γ)))_+
```

Where:
- `H_ik`: Output at position i, dimension k
- `V_jk`: Value at position j, dimension k
- `Q_i`, `K_j`: Query and key vectors
- `||Q_i - K_j||²_2`: Squared L2 norm (Euclidean distance squared)
- `d`: Dimension of the attention head
- `γ`: Scaling parameter (initialized to sqrt(head_dimension))
- `()_+`: ReLU function (approximated in the encrypted domain)

### CKKS Parameters

The implementation uses the CKKS encryption scheme with the following default parameters:
- Polynomial modulus degree: 65536
- Coefficient modulus: Default values for 5 levels
- Scale: 2^40 (provides good balance between precision and performance)

### Approximations

Due to the constraints of homomorphic encryption, several approximations are used:
- ReLU activation: Approximated by a polynomial function
- Layer normalization: Simplified in the encrypted domain

## Performance Considerations

- The performance is heavily dependent on the GPU hardware, CKKS parameters, and model size
- Larger models (more layers, larger hidden dimensions) will require more computational resources
- Reducing the polynomial modulus degree can improve performance at the cost of precision
- The implementation includes timing measurements to help analyze performance

## Security Considerations

This implementation provides computational security based on the security of the CKKS scheme in HEonGPU. The security level depends on the chosen polynomial modulus degree and coefficient modulus bit lengths.

## Future Improvements

- Implement more efficient polynomial approximations for non-linear functions
- Add support for batch processing of multiple inputs
- Optimize memory usage and computational performance
- Extend to support other attention mechanisms

## License

This project is licensed under the same terms as HEonGPU. 