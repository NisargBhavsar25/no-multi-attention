#!/usr/bin/env python3
import os
import argparse
import subprocess
import numpy as np
import torch
from transformers import AutoTokenizer
import struct
import tempfile
import logging

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

def convert_model_weights(model_dir, output_dir):
    """Convert pre-converted weights to SEAL format if needed."""
    # The model is already converted, just check if it exists
    required_files = ['wq.bin', 'wk.bin', 'wv.bin', 'wo.bin', 'ff1.bin', 'ff2.bin']
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            raise FileNotFoundError(f"Required weight file {file} not found in {model_dir}")
    
    # Copy the files to the output directory if it's different
    if model_dir != output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for file in required_files:
            src_path = os.path.join(model_dir, file)
            dst_path = os.path.join(output_dir, file)
            with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
                dst.write(src.read())
    
    logging.info(f"Model weights prepared in {output_dir}")
    return output_dir

def tokenize_text(text, max_seq_length=128):
    """Tokenize input text using BERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the input text
    tokens = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_tensors='pt'
    )
    
    logging.info(f"Tokenized text into {tokens['input_ids'].shape[1]} tokens")
    
    # Get the attention mask
    attention_mask = tokens['attention_mask']
    
    # Get the input IDs
    input_ids = tokens['input_ids']
    
    return input_ids, attention_mask

def get_embeddings(input_ids):
    """Convert input IDs to embeddings.
    
    For simplicity, we'll use random embeddings of the right size instead 
    of loading the full BERT model just for embeddings.
    """
    # In a real implementation, we would load the embeddings from the trained model
    # Here we'll just use random values of the right shape for demonstration
    seq_length = input_ids.shape[1]
    hidden_size = 768  # Standard BERT hidden size
    
    # Generate random embeddings (in a real scenario, we'd use the actual embeddings)
    embeddings = torch.randn(1, seq_length, hidden_size)
    
    logging.info(f"Created embeddings with shape {embeddings.shape}")
    return embeddings

def prepare_input_file(embeddings, attention_mask, output_dir):
    """Create binary input file for the SEAL implementation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten embeddings and convert to numpy
    embeddings_flat = embeddings.numpy().flatten()
    
    # Convert attention mask to numpy
    attention_mask_np = attention_mask.numpy()
    
    # Write embeddings to binary file
    embeddings_path = os.path.join(output_dir, 'input_embeddings.bin')
    with open(embeddings_path, 'wb') as f:
        # Write number of items
        f.write(struct.pack('I', len(embeddings_flat)))
        
        # Write embedding values as doubles
        for val in embeddings_flat:
            f.write(struct.pack('d', float(val)))
    
    # Write attention mask to binary file
    mask_path = os.path.join(output_dir, 'attention_mask.bin')
    with open(mask_path, 'wb') as f:
        # Write dimensions
        seq_length = attention_mask_np.shape[1]
        f.write(struct.pack('I', seq_length))
        
        # Write mask values as bytes (0 or 1)
        for val in attention_mask_np.flatten():
            f.write(struct.pack('B', int(val)))
    
    logging.info(f"Prepared input files in {output_dir}")
    return embeddings_path, mask_path

def run_encrypted_inference(model_dir, input_file, mask_file, output_file, poly_modulus_degree=8192, num_layers=1, hidden_size=768, num_attention_heads=12):
    """Run the encrypted inference using the SEAL implementation."""
    # Construct and run the command to execute the SEAL binary
    cmd = [
        "./encrypted_SEAL/build/bin/encrypted_seal_transformer",
        "--model_path", model_dir,
        "--input_file", input_file,
        "--mask_file", mask_file, 
        "--output_file", output_file,
        "--poly_modulus_degree", str(poly_modulus_degree),
        "--num_layers", str(num_layers),
        "--hidden_size", str(hidden_size),
        "--num_attention_heads", str(num_attention_heads)
    ]
    
    logging.info(f"Running encrypted inference with command: {' '.join(cmd)}")
    
    try:
        # Execute the command
        process = subprocess.run(
            cmd, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Log output
        logging.info("Encrypted inference completed successfully")
        logging.debug(process.stdout)
        
        if process.stderr:
            logging.warning(f"Stderr: {process.stderr}")
            
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Encrypted inference failed: {e}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        return False

def read_output_file(output_file):
    """Read and parse the output file from the SEAL implementation."""
    if not os.path.exists(output_file):
        logging.error(f"Output file {output_file} not found")
        return None
    
    try:
        with open(output_file, 'rb') as f:
            # Read number of values
            num_values = struct.unpack('I', f.read(4))[0]
            
            # Read the values
            values = []
            for _ in range(num_values):
                value = struct.unpack('d', f.read(8))[0]
                values.append(value)
            
            return np.array(values)
    except Exception as e:
        logging.error(f"Error reading output file: {e}")
        return None

def interpret_results(output_values, tokenizer):
    """Interpret the output values from the encrypted inference."""
    if output_values is None:
        return "Failed to process results"
    
    # In a real implementation, we would do proper interpretation based on the task
    # For demonstration, we'll just report the statistics of the output
    
    mean_val = np.mean(output_values)
    min_val = np.min(output_values)
    max_val = np.max(output_values)
    
    result = (
        f"Processed sentence in encrypted domain.\n"
        f"Output statistics:\n"
        f"  Mean: {mean_val:.4f}\n"
        f"  Min: {min_val:.4f}\n"
        f"  Max: {max_val:.4f}\n"
    )
    
    return result

def process_sentence(sentence, model_dir='converted_model_quadratic_relu', 
                     temp_dir=None, poly_modulus_degree=8192, num_layers=1, 
                     hidden_size=768, num_attention_heads=12):
    """Process a sentence through the encrypted transformer."""
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    try:
        # Step 1: Tokenize the sentence
        input_ids, attention_mask = tokenize_text(sentence)
        
        # Step 2: Convert to embeddings
        embeddings = get_embeddings(input_ids)
        
        # Step 3: Prepare input files
        input_file, mask_file = prepare_input_file(embeddings, attention_mask, temp_dir)
        
        # Step 4: Set up output file path
        output_file = os.path.join(temp_dir, 'output.bin')
        
        # Step 5: Ensure model weights are properly converted
        model_dir = convert_model_weights(model_dir, os.path.join(temp_dir, 'model'))
        
        # Step 6: Run encrypted inference
        success = run_encrypted_inference(
            model_dir, 
            input_file, 
            mask_file, 
            output_file, 
            poly_modulus_degree,
            num_layers,
            hidden_size,
            num_attention_heads
        )
        
        if not success:
            return "Encrypted inference failed. See logs for details."
        
        # Step 7: Read output file
        output_values = read_output_file(output_file)
        
        # Step 8: Interpret results
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        result = interpret_results(output_values, tokenizer)
        
        return result
    
    except Exception as e:
        logging.error(f"Error processing sentence: {e}", exc_info=True)
        return f"Error: {str(e)}"

def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(description='Process sentences through encrypted quadratic inhibitor attention model.')
    parser.add_argument('--sentence', type=str, required=True, help='Sentence to process')
    parser.add_argument('--model_dir', type=str, default='converted_model_quadratic_relu', 
                        help='Directory containing the converted model weights')
    parser.add_argument('--temp_dir', type=str, default=None, 
                        help='Temporary directory for intermediate files')
    parser.add_argument('--poly_modulus_degree', type=int, default=8192, 
                        help='Polynomial modulus degree for SEAL')
    parser.add_argument('--num_layers', type=int, default=1, 
                        help='Number of transformer layers')
    parser.add_argument('--hidden_size', type=int, default=768, 
                        help='Hidden size of the model')
    parser.add_argument('--num_attention_heads', type=int, default=12, 
                        help='Number of attention heads')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Process the sentence
    result = process_sentence(
        args.sentence,
        args.model_dir,
        args.temp_dir,
        args.poly_modulus_degree,
        args.num_layers,
        args.hidden_size,
        args.num_attention_heads
    )
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main() 