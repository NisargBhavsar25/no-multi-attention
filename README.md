# BERT Transformer Research: Attention Mechanisms and Encrypted Inference

This repository contains implementations of different BERT transformer attention mechanisms and a privacy-preserving encrypted inference pipeline using homomorphic encryption.

## Overview

This project investigates three different attention mechanisms in BERT-like transformer models:

1. **Standard Attention** - The original scaled dot-product attention
2. **Inhibitor Attention** - Manhattan distance-based attention with ReLU inhibition
3. **Quadratic Inhibitor Attention** - Quadratic approximation compatible with homomorphic encryption

The repository also includes an implementation of an encrypted inference pipeline using Microsoft SEAL for privacy-preserving sentiment analysis.

## Project Structure

```
.
├── data/                      # Dataset storage
├── models/                    # Different transformer implementations
│   ├── attention/             # Various attention mechanisms
│   ├── embeddings/            # Different embedding implementations
│   └── layers/                # Custom transformer layers
├── experiments/               # Experiment configurations and results
├── utils/                     # Utility functions
├── notebooks/                 # Jupyter notebooks for analysis
├── encrypted_SEAL/            # Homomorphic encryption implementation
│   ├── build/                 # Build directory for C++ code
│   └── ...                    # SEAL-based transformer implementation
└── run_seal.py                # Python interface for encrypted inference
```

## Encrypted Inference with Microsoft SEAL

The `encrypted_SEAL` directory contains a C++ implementation of a transformer with quadratic inhibitor attention using Microsoft SEAL for homomorphic encryption. This allows inference on encrypted data, preserving privacy throughout the process.

### Recent Updates to Encrypted Pipeline

- **Simplified Pipeline**: Implemented a robust attention and feed-forward mechanism for parameter compatibility
- **Parameter Handling**: Added error handling and fallback mechanisms for parameter mismatches
- **Python Interface**: Enhanced `run_seal.py` script for easy inference with BERT tokenization
- **Timing Information**: Added timing measurements for overall inference process
- **Adaptive Scaling**: Implemented dynamic scale matching for compatibility between operations

### Running Encrypted Inference

To run the encrypted inference on a sample sentence:

```bash
python run_seal.py --sentence "This movie was great, I really enjoyed it!" --poly_modulus_degree 32768 --hidden_size 64 --num_heads 2
```

Command line options:
- `--sentence`: Input text to process
- `--poly_modulus_degree`: CKKS polynomial modulus degree (default: 131072)
- `--hidden_size`: Hidden dimension size (default: inferred from model)
- `--num_heads`: Number of attention heads (default: inferred from model)
- `--seq_length`: Maximum sequence length (default: 16)

### Building the Encrypted Implementation

To build the C++ encrypted transformer implementation:

```bash
cd encrypted_SEAL
mkdir build
cd build
cmake ..
make
```

## Running Attention Comparison Experiments

The repository contains code to systematically compare different attention mechanisms.

### Full Experiment

The full experiment trains and evaluates each model on the complete IMDB dataset. To run:

```bash
python run_experiment.py --config experiments/attention_comparison_config.yaml
```

### Hardware Requirements for Full Experiment

- GPU with at least 8GB VRAM
- At least 16GB RAM
- About 20GB free disk space for models and datasets

### Configuration Details

The full experiment uses the following configuration:
- Model size: 768 hidden dimensions
- 6 transformer layers
- 12 attention heads
- Sequence length: 256 tokens
- 3 training epochs
- Batch size: 16
- Complete IMDB dataset (25,000 training samples, 25,000 test samples)

### Running Individual Models

Train or evaluate a specific attention mechanism:

```bash
# Training
python train.py --config experiments/attention_comparison_config.yaml --attention_type standard
python train.py --config experiments/attention_comparison_config.yaml --attention_type inhibitor
python train.py --config experiments/attention_comparison_config.yaml --attention_type quadratic_inhibitor

# Evaluation
python evaluate.py --model_path experiments/results/attention_comparison_full/standard_attention/final_model --config experiments/attention_comparison_config.yaml --attention_type standard
```

## Setup for Experiments

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Experiment Metrics

The experiments track the following metrics:
- Accuracy on IMDB test set
- Training time
- Inference latency
- Model size
- Memory usage

Results are stored in the `experiments/results` directory, with detailed analysis in the notebooks.

## Encrypted Implementation Performance

- Total inference time (including Python overhead): ~8-10 seconds for short sentences
- C++ core inference time: ~0.1-0.2 seconds for the encryption operations
- Performance can be improved by:
  - Reducing polynomial modulus degree for less demanding applications
  - Using custom weight configurations optimized for HE operations
  - Implementing parallel processing for batch inference

## Security and Privacy

The homomorphic encryption implementation provides computational security based on the CKKS scheme in Microsoft SEAL. This allows for privacy-preserving inference where the input data remains encrypted throughout the computation process.

## Future Work

- Implement more efficient polynomial approximations for non-linear functions
- Add support for batch processing of multiple inputs
- Optimize memory usage and computational performance
- Explore hybrid approaches combining CPU and GPU operations
- Extend to support other attention mechanisms and model architectures

## License

This project is open source, with the encrypted components licensed under the same terms as Microsoft SEAL (MIT License). 