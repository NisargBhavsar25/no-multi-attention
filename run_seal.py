#!/usr/bin/env python3
import os
import subprocess
import tempfile
import shutil
import numpy as np
import struct
import argparse
from transformers import AutoTokenizer

def prepare_input_data(sentence, seq_length=16, hidden_size=64):
    """Prepare input data for SEAL inference"""
    print(f"Preparing input for: '{sentence}'")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Step 1: Tokenize with BERT
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer(
        sentence, 
        padding='max_length', 
        truncation=True, 
        max_length=seq_length, 
        return_tensors='pt'
    )
    
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    
    # Print token information
    token_strings = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"Tokenized into {sum(attention_mask[0])} tokens (out of {len(token_strings)} padded)")
    
    # Step 2: Get embeddings from BERT model
    from transformers import BertModel
    import torch
    model = BertModel.from_pretrained('bert-base-uncased')
    tokens_tensor = input_ids
    embeddings = model.bert.embeddings(tokens_tensor, torch.tensor([[1] * len(token_strings)]))
    embeddings = embeddings.detach().numpy()
    
    # Step 3: Write embeddings to binary file
    embeddings_flat = embeddings.flatten()
    embeddings_path = os.path.join(temp_dir, "input_embeddings.bin")
    
    with open(embeddings_path, 'wb') as f:
        # Write number of elements
        f.write(struct.pack('I', len(embeddings_flat)))
        
        # Write values
        for val in embeddings_flat:
            f.write(struct.pack('d', float(val)))
    
    # Step 4: Write attention mask
    mask_path = os.path.join(temp_dir, "attention_mask.bin")
    with open(mask_path, 'wb') as f:
        # Write sequence length
        f.write(struct.pack('I', seq_length))
        
        # Write mask values
        for val in attention_mask.numpy().flatten():
            f.write(struct.pack('B', int(val)))
    
    # Step 5: Create output path
    output_path = os.path.join(temp_dir, "output.bin")
    
    return {
        'temp_dir': temp_dir,
        'input_path': embeddings_path,
        'mask_path': mask_path,
        'output_path': output_path
    }

def run_seal_inference(data_paths, model_dir, poly_modulus_degree=8192, 
                      hidden_size=64, num_attention_heads=2, num_layers=1,
                      seal_binary_path="./bin/encrypted_seal_transformer"):
    """Run the SEAL inference"""
    
    # Build command
    cmd = [
        seal_binary_path,
        "--model_path", model_dir,
        "--input_file", data_paths['input_path'],
        "--mask_file", data_paths['mask_path'],
        "--output_file", data_paths['output_path'],
        "--poly_modulus_degree", str(poly_modulus_degree),
        "--hidden_size", str(hidden_size),
        "--num_attention_heads", str(num_attention_heads),
        "--num_layers", str(num_layers)
    ]
    
    print("\nRunning SEAL inference with command:")
    print(" ".join(cmd))
    
    # Execute the command
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print("Inference completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def read_output(output_path):
    """Read and interpret the output file"""
    if not os.path.exists(output_path):
        print(f"Error: Output file not found at {output_path}")
        return None
    
    try:
        with open(output_path, 'rb') as f:
            # Read number of values
            num_values = struct.unpack('I', f.read(4))[0]
            
            # Read values
            values = []
            for _ in range(num_values):
                value = struct.unpack('d', f.read(8))[0]
                values.append(value)
        
        return np.array(values)
    except Exception as e:
        print(f"Error reading output file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Run SEAL inference on a sentence')
    parser.add_argument('--sentence', type=str, default="This movie was great, I really enjoyed it!",
                       help='Sentence to process')
    parser.add_argument('--model_dir', type=str, default="../../converted_model_quadratic_relu",
                       help='Path to the model weights')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='Model hidden size (must be reduced from original)')
    parser.add_argument('--num_heads', type=int, default=2,
                       help='Number of attention heads (must be reduced from original)')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Number of transformer layers (must be reduced from original)')
    parser.add_argument('--seq_length', type=int, default=16,
                       help='Sequence length')
    parser.add_argument('--poly_modulus_degree', type=int, default=8192,
                       help='Polynomial modulus degree for SEAL')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("RUNNING ACTUAL SEAL INFERENCE PIPELINE")
    print("=" * 70)
    print(f"Using reduced model parameters due to encryption constraints:")
    print(f"  hidden_size: {args.hidden_size} (original: 384)")
    print(f"  num_heads: {args.num_heads} (original: 6)")
    print(f"  layers: {args.num_layers} (original: 3)")
    print(f"  poly_modulus_degree: {args.poly_modulus_degree}")
    print()
    
    # Prepare input data
    data_paths = prepare_input_data(
        args.sentence, 
        seq_length=args.seq_length,
        hidden_size=args.hidden_size
    )
    
    # Run inference
    success = run_seal_inference(
        data_paths,
        args.model_dir,
        poly_modulus_degree=args.poly_modulus_degree,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    # Process output if successful
    if success:
        output_values = read_output(data_paths['output_path'])
        
        if output_values is not None:
            print("\nInference Results:")
            print(f"  Output size: {len(output_values)} values")
            print(f"  Mean: {np.mean(output_values):.6f}")
            print(f"  Min: {np.min(output_values):.6f}")
            print(f"  Max: {np.max(output_values):.6f}")
            
            # If output matches expected size, reshape it
            expected_size = args.seq_length * args.hidden_size
            if len(output_values) == expected_size:
                reshaped = output_values.reshape(args.seq_length, args.hidden_size)
                cls_score = np.mean(reshaped[0])
                print(f"\nSentiment score: {cls_score:.6f}")
                sentiment = "Positive" if cls_score > 0 else "Negative"
                print(f"Detected sentiment: {sentiment}")
    
    # Clean up
    shutil.rmtree(data_paths['temp_dir'])
    print("\nCleanup complete. Temporary files removed.")

if __name__ == "__main__":
    main() 