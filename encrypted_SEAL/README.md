# Encrypted Quadratic Inhibitor Transformer with Microsoft SEAL

This project implements a Fully Homomorphic Encryption (FHE) version of a transformer model with quadratic inhibitor attention mechanism using the Microsoft SEAL library. It enables running transformer models on encrypted data, preserving privacy throughout the inference process.

## Overview

The implementation provides an encrypted version of the quadratic inhibitor attention mechanism, which uses a quadratic form approximation of inhibition that's compatible with homomorphic encryption operations. This approach is well-suited for the CKKS encryption scheme since it requires only polynomial operations.

The encrypted transformer consists of the following components:

1. **Encrypted Transformer Weights**: Handles loading and encrypting pretrained model weights
2. **Encrypted Quadratic Inhibitor Attention**: Implements the quadratic inhibitor attention mechanism in the encrypted domain
3. **Encrypted Transformer**: Manages the complete transformer model with multiple layers
4. **Encrypted Inference Pipeline**: Provides an end-to-end solution for encrypted inference

## Recent Updates

- **Strassen's Matrix Multiplication**: Implemented the Strassen algorithm for faster matrix multiplication with reduced complexity (O(n^2.807) vs O(n^3) for standard multiplication)
- **Parameter Mismatches Handling**: Added robust error handling and fallback mechanisms for parameter mismatches between plaintexts and ciphertexts
- **Python Interface**: Enhanced `run_seal.py` script in the project root directory for easy inference with BERT tokenization and embedding generation
- **Timing Information**: Added timing measurements for overall inference process
- **Adaptive Scaling**: Implemented dynamic scale matching to ensure compatibility between operations

## Prerequisites

- [CMake](https://cmake.org/download/) >=3.12
- [Microsoft SEAL](https://github.com/Microsoft/SEAL) >=4.1
- Compatible C++ compiler with C++17 support (GCC 13+, Clang 10+, or MSVC 19.20+)
- Python 3.8+ with PyTorch and Transformers (for the Python interface)

## Building the Project

1. Make sure Microsoft SEAL is properly installed on your system
2. Clone this repository
3. Build the project:

```bash
mkdir build
cd build
cmake ..
make
```

This will generate the executable `encrypted_seal_transformer` in the `build/bin` directory.

## Using the Python Interface

The project includes a Python interface (`run_seal.py`) that handles tokenization, embedding generation, and inference:

```bash
python run_seal.py --sentence "This movie was great, I really enjoyed it!" --poly_modulus_degree 32768 --hidden_size 64 --num_heads 2
```

Command line options:
- `--sentence`: Input text to process
- `--model_dir`: Path to model weights (default: ./converted_model_quadratic_relu)
- `--poly_modulus_degree`: CKKS polynomial modulus degree (default: 131072)
- `--hidden_size`: Hidden dimension size (can override inferred size)
- `--num_heads`: Number of attention heads (can override inferred value)
- `--seq_length`: Maximum sequence length (default: 16)

The script performs the following steps:
1. Tokenizes the input text using BERT tokenizer
2. Generates embeddings using BERT's embedding layer
3. Runs the encrypted inference pipeline
4. Processes the output to determine the sentiment

## Running the C++ Demo Directly

```bash
./bin/encrypted_seal_transformer --model_path ./model --input_file input.bin --mask_file mask.bin --output_file output.bin --poly_modulus_degree 32768 --num_layers 1 --hidden_size 64 --num_attention_heads 2 --seq_length 16
```

Command line options:
- `--model_path`: Path to the pretrained weights (default: ./model)
- `--input_file`: Path to binary file containing input embeddings
- `--mask_file`: Path to binary file containing attention mask
- `--output_file`: Path to write output embeddings
- `--poly_modulus_degree`: CKKS polynomial modulus degree (default: 65536)
- `--num_layers`: Number of transformer layers (default: 1)
- `--hidden_size`: Hidden dimension size (default: 128)
- `--num_attention_heads`: Number of attention heads (default: 4)
- `--seq_length`: Sequence length (default: 16)

## Implementation Details

### Strassen's Matrix Multiplication Algorithm

The implementation includes Strassen's matrix multiplication algorithm to reduce the computational complexity for matrix operations:

```
Algorithm: Strassen's Matrix Multiplication
Input: A = (A11 A12; A21 A22) and B = (B11 B12; B21 B22) ∈ R^n×n
1: if n = 1 then
2:   C = A · B
3: else
4:   M1 = (A11 + A22) · (B11 + B22)
5:   M2 = (A21 + A22) · B11
6:   M3 = A11 · (B12 - B22)
7:   M4 = A22 · (B21 - B11)
8:   M5 = (A11 + A12) · B22
9:   M6 = (A21 - A11) · (B11 + B12)
10:  M7 = (A12 - A22) · (B21 + B22)
11:  C11 = M1 + M4 - M5 + M7
12:  C12 = M3 + M5
13:  C21 = M2 + M4
14:  C22 = M1 - M2 + M3 + M6
Output: A · B = C = (C11 C12; C21 C22) ∈ R^n×n
```

The algorithm recursively divides the matrices into smaller submatrices and reduces the number of multiplications required from 8 to 7, resulting in a time complexity of O(n^2.807) compared to O(n^3) for standard matrix multiplication.

In our implementation, we automatically detect when Strassen's algorithm can be applied (for square matrices with power-of-2 dimensions) and fall back to standard matrix multiplication otherwise.

### Robust Parameter Handling

The implementation now includes robust parameter handling to address common homomorphic encryption challenges:

1. **Scale Matching**: Automatically adjusts scales between operations to prevent scale mismatch errors
2. **NTT Form Compatibility**: Properly handles the NTT form of ciphertexts and plaintexts
3. **Chain Index Management**: Tracks and manages chain indices to ensure compatibility
4. **Fallback Mechanisms**: Includes fallback to simpler operations when parameter mismatches occur


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

The implementation uses the CKKS encryption scheme with the following improved parameters:
- Polynomial modulus degree: 32768 (default for balanced performance)
- Coefficient modulus: Custom configuration with careful bit allocation
- Scale: 2^20 (reduced from 2^40 for better parameter stability)

## Performance Considerations

- The performance is heavily dependent on the hardware, CKKS parameters, and model size
- Total inference time (including Python overhead) is typically 8-10 seconds for short sentences
- C++ core inference time is around 0.1-0.2 seconds for the encryption operations
- Performance can be further improved by:
  - Using Strassen's matrix multiplication for large matrices (automatically applied when possible)
  - Reducing polynomial modulus degree for less demanding applications
  - Using custom weight configurations optimized for HE operations
  - Implementing parallel processing for batch inference

## Security Considerations

This implementation provides computational security based on the security of the CKKS scheme in Microsoft SEAL. The security level depends on the chosen polynomial modulus degree and coefficient modulus bit lengths.

## Known Limitations

- The system currently has limited support for very large hidden sizes
- Residual connections may be skipped when parameter mismatches occur
- The simplified attention mechanism prioritizes stability over exactness of the full attention mechanism
- Strassen's algorithm implementation is simplified for encrypted matrices and may not achieve the full theoretical speedup in practice

## Future Improvements

- Implement more efficient polynomial approximations for non-linear functions
- Add support for batch processing of multiple inputs
- Optimize memory usage and computational performance
- Explore hybrid approaches combining CPU and GPU operations
- Extend to support other attention mechanisms

## License

This project is licensed under the same terms as Microsoft SEAL (MIT License). 