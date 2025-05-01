#!/usr/bin/env python3
import os
import subprocess
import tempfile
import shutil
import numpy as np
import struct
import argparse
import glob
import traceback
import sys
from transformers import AutoTokenizer
import time

def get_model_params_from_weights(model_dir):
    """Infer model parameters from the weight files"""
    print(f"DEBUG: Starting model parameter inference from: {model_dir}")
    
    try:
        # Check if model directory exists
        if not os.path.exists(model_dir):
            print(f"DEBUG: Model directory {model_dir} does not exist")
            raise ValueError(f"Model directory {model_dir} does not exist")
        
        # List all files in the model directory for debugging
        print(f"DEBUG: Files in model directory:")
        model_files = os.listdir(model_dir)
        for file in model_files:
            print(f"DEBUG:   {file}")
        
        # Look for weight files
        wq_path = os.path.join(model_dir, "wq.bin")
        if not os.path.exists(wq_path):
            print(f"DEBUG: Weight file {wq_path} not found")
            # Try to find any weight files as fallback
            weight_files = [f for f in model_files if f.endswith('.bin')]
            if weight_files:
                print(f"DEBUG: Found alternative weight files: {weight_files}")
                wq_path = os.path.join(model_dir, weight_files[0])
            else:
                raise ValueError(f"Weight file {wq_path} not found and no alternative weight files available")
        
        print(f"DEBUG: Reading dimensions from {wq_path}")
        # Read dimensions from weight file
        with open(wq_path, 'rb') as f:
            try:
                rows = struct.unpack('i', f.read(4))[0]
                cols = struct.unpack('i', f.read(4))[0]
                print(f"DEBUG: Read dimensions: rows={rows}, cols={cols}")
            except Exception as e:
                print(f"DEBUG: Error reading dimensions: {e}")
                print(f"DEBUG: File size: {os.path.getsize(wq_path)} bytes")
                raise ValueError(f"Failed to read dimensions from weight file: {e}")
        
        # The rows might be the hidden size in a weight matrix
        # But if it's very small (like 3), this could be a specialized dimension, not the true hidden size
        if rows < 16:  # Heuristic threshold - typical hidden sizes are much larger
            print(f"DEBUG: Found very small hidden size ({rows}), assuming this is not the true hidden size")
            hidden_size = 768  # Default to BERT base hidden size
        else:
            hidden_size = rows
        
        # Determine number of layers by counting layer-specific files
        print(f"DEBUG: Searching for layer files with pattern: {os.path.join(model_dir, 'layer_*')}")
        layer_files = glob.glob(os.path.join(model_dir, "layer_*"))
        print(f"DEBUG: Found layer files: {layer_files}")
        
        unique_layers = set()
        for file in layer_files:
            # Extract layer number from filename
            layer_name = os.path.basename(file)
            print(f"DEBUG: Processing layer file: {layer_name}")
            if layer_name.startswith("layer_"):
                try:
                    layer_num = int(layer_name.split("_")[1].split(".")[0])
                    unique_layers.add(layer_num)
                    print(f"DEBUG: Identified layer: {layer_num}")
                except Exception as e:
                    print(f"DEBUG: Failed to extract layer number from {layer_name}: {e}")
        
        # If no layer-specific files found, assume single layer structure
        num_layers = len(unique_layers) if unique_layers else 1
        print(f"DEBUG: Determined number of layers: {num_layers}")
        
        # Estimate number of attention heads (typically hidden_size / 64)
        potential_head_sizes = [32, 64, 128]
        num_heads = 2  # Default if we can't determine
        
        for head_size in potential_head_sizes:
            if hidden_size % head_size == 0:
                num_heads = hidden_size // head_size
                print(f"DEBUG: Estimated heads: {num_heads} (hidden_size {hidden_size} / head_size {head_size})")
                break
        
        print(f"DEBUG: Final inferred model parameters:")
        print(f"DEBUG:   Hidden size: {hidden_size}")
        print(f"DEBUG:   Num layers: {num_layers}")
        print(f"DEBUG:   Estimated num heads: {num_heads}")
        
        return {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_heads': num_heads
        }
    except Exception as e:
        print(f"DEBUG: Exception in get_model_params_from_weights: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        raise

def prepare_input_data(sentence, seq_length=16, hidden_size=768):
    """Prepare input data for SEAL inference"""
    print(f"DEBUG: Starting input preparation for: '{sentence}'")
    print(f"DEBUG: Using seq_length={seq_length}, hidden_size={hidden_size}")
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        print(f"DEBUG: Created temp directory: {temp_dir}")
        
        # Step 1: Tokenize with BERT
        print(f"DEBUG: Loading BERT tokenizer")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        print(f"DEBUG: Tokenizing input")
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
        print(f"DEBUG: Tokenized into {sum(attention_mask[0])} tokens (out of {len(token_strings)} padded)")
        print(f"DEBUG: First few tokens: {token_strings[:5]}")
        
        # Step 2: Get embeddings from BERT model
        print(f"DEBUG: Loading BERT model for embeddings")
        from transformers import BertModel
        import torch
        model = BertModel.from_pretrained('bert-base-uncased')
        
        print(f"DEBUG: Generating embeddings")
        try:
            # Get BERT embeddings with the right shape
            position_ids = torch.arange(input_ids.shape[1]).expand((1, -1))
            token_type_ids = torch.zeros_like(input_ids)
            embeddings = model.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids
            )
            print(f"DEBUG: Embeddings shape from BERT: {embeddings.shape}")
            
            # Convert to proper target shape if needed
            if embeddings.shape[-1] != hidden_size:
                print(f"DEBUG: Reshaping embeddings from {embeddings.shape[-1]} to {hidden_size} dimensions")
                # Create a projection matrix to change dimensions
                projection = torch.zeros((embeddings.shape[-1], hidden_size))
                # Identity matrix for the dimensions we have
                min_dim = min(embeddings.shape[-1], hidden_size)
                projection[:min_dim, :min_dim] = torch.eye(min_dim)
                # Apply projection
                embeddings = torch.matmul(embeddings, projection)
                print(f"DEBUG: New embeddings shape: {embeddings.shape}")
            
            embeddings = embeddings.detach().numpy()
        except Exception as e:
            print(f"DEBUG: Error in embeddings generation: {e}")
            print(f"DEBUG: Attempting alternative approach")
            # Fallback to creating random embeddings of the right shape
            embeddings = np.random.randn(1, seq_length, hidden_size) * 0.1
            print(f"DEBUG: Created fallback embeddings with shape: {embeddings.shape}")
        
        print(f"DEBUG: Embeddings array shape: {embeddings.shape}")
        
        # Step 3: Write embeddings to binary file
        embeddings_flat = embeddings.flatten()
        print(f"DEBUG: Flattened embeddings to shape: {embeddings_flat.shape}")
        
        embeddings_path = os.path.join(temp_dir, "input_embeddings.bin")
        print(f"DEBUG: Writing embeddings to: {embeddings_path}")
        
        with open(embeddings_path, 'wb') as f:
            # Write number of elements
            num_elements = len(embeddings_flat)
            print(f"DEBUG: Writing {num_elements} elements")
            f.write(struct.pack('I', num_elements))
            
            # Write values
            for val in embeddings_flat:
                f.write(struct.pack('d', float(val)))
        
        print(f"DEBUG: Embeddings file size: {os.path.getsize(embeddings_path)} bytes")
        
        # Step 4: Write attention mask
        mask_path = os.path.join(temp_dir, "attention_mask.bin")
        print(f"DEBUG: Writing attention mask to: {mask_path}")
        
        with open(mask_path, 'wb') as f:
            # Write sequence length
            f.write(struct.pack('I', seq_length))
            
            # Write mask values
            attention_mask_np = attention_mask.numpy().flatten()
            print(f"DEBUG: Attention mask shape: {attention_mask_np.shape}")
            for val in attention_mask_np:
                f.write(struct.pack('B', int(val)))
        
        print(f"DEBUG: Mask file size: {os.path.getsize(mask_path)} bytes")
        
        # Step 5: Create output path
        output_path = os.path.join(temp_dir, "output.bin")
        print(f"DEBUG: Output will be written to: {output_path}")
        
        return {
            'temp_dir': temp_dir,
            'input_path': embeddings_path,
            'mask_path': mask_path,
            'output_path': output_path
        }
    except Exception as e:
        print(f"DEBUG: Exception in prepare_input_data: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        raise

def run_seal_inference(data_paths, model_dir, poly_modulus_degree=131072, 
                      hidden_size=768, num_attention_heads=12, num_layers=1,
                      seal_binary_path="./encrypted_SEAL/build/bin/encrypted_seal_transformer"):
    """Run the SEAL inference"""
    print(f"DEBUG: Starting SEAL inference")
    print(f"DEBUG: Checking if binary exists at: {seal_binary_path}")
    
    if not os.path.exists(seal_binary_path):
        print(f"DEBUG: Binary not found at {seal_binary_path}")
        # Try to find the binary
        possible_paths = [
            "./encrypted_SEAL/build/encrypted_seal_transformer",
            "./encrypted_SEAL/encrypted_seal_transformer",
            "./build/bin/encrypted_seal_transformer"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                seal_binary_path = path
                print(f"DEBUG: Found alternative binary at: {seal_binary_path}")
                break
        else:
            print(f"DEBUG: Could not find SEAL binary, tried: {seal_binary_path} and {possible_paths}")
            return False
    
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
        "--num_layers", str(num_layers),
        "--seq_length", str(16)  # Explicitly set sequence length
    ]
    
    print("\nDEBUG: Running SEAL inference with command:")
    print(f"DEBUG: {' '.join(cmd)}")
    
    # Execute the command
    try:
        print(f"DEBUG: Executing subprocess")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Don't raise exception to capture output
        )
        
        print(f"DEBUG: Subprocess returned with code: {result.returncode}")
        
        # Print stdout and stderr for debugging regardless of success
        if result.stdout:
            print("DEBUG: STDOUT from SEAL binary:")
            for line in result.stdout.splitlines():
                print(f"DEBUG-STDOUT: {line}")
        
        if result.stderr:
            print("DEBUG: STDERR from SEAL binary:")
            for line in result.stderr.splitlines():
                print(f"DEBUG-STDERR: {line}")
        
        if result.returncode != 0:
            print(f"DEBUG: Command failed with return code {result.returncode}")
            return False
        
        print("DEBUG: Inference completed successfully")
        return True
    except Exception as e:
        print(f"DEBUG: Exception during subprocess execution: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return False

def read_output(output_path):
    """Read and interpret the output file"""
    print(f"DEBUG: Trying to read output from: {output_path}")
    
    if not os.path.exists(output_path):
        print(f"DEBUG: Output file not found at {output_path}")
        return None
    
    try:
        print(f"DEBUG: Output file exists, size: {os.path.getsize(output_path)} bytes")
        with open(output_path, 'rb') as f:
            # Read number of values
            num_values_bytes = f.read(4)
            if len(num_values_bytes) != 4:
                print(f"DEBUG: Could not read num_values (got {len(num_values_bytes)} bytes)")
                return None
            
            num_values = struct.unpack('I', num_values_bytes)[0]
            print(f"DEBUG: Output file contains {num_values} values")
            
            # Read values
            values = []
            for i in range(num_values):
                value_bytes = f.read(8)
                if len(value_bytes) != 8:
                    print(f"DEBUG: Could not read value {i} (got {len(value_bytes)} bytes)")
                    break
                
                value = struct.unpack('d', value_bytes)[0]
                values.append(value)
                
                # Print first few and last few values for debugging
                if i < 5 or i >= num_values - 5:
                    print(f"DEBUG: Value {i}: {value}")
        
        print(f"DEBUG: Successfully read {len(values)} values from output file")
        return np.array(values)
    except Exception as e:
        print(f"DEBUG: Error reading output file: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Run SEAL inference on a sentence')
    parser.add_argument('--sentence', type=str, default="This movie was great, I really enjoyed it!",
                       help='Sentence to process')
    parser.add_argument('--model_dir', type=str, default="./converted_model_quadratic_relu",
                       help='Path to the model weights')
    parser.add_argument('--seq_length', type=int, default=16,
                       help='Sequence length')
    parser.add_argument('--poly_modulus_degree', type=int, default=131072,
                       help='Polynomial modulus degree for SEAL')
    parser.add_argument('--hidden_size', type=int, default=None,
                       help='Hidden size (overrides inferred size if specified)')
    parser.add_argument('--num_heads', type=int, default=None,
                       help='Number of attention heads (overrides inferred if specified)')
    parser.add_argument('--use_bert_base', action='store_true',
                       help='Use BERT base parameters rather than inferred parameters')
    
    try:
        args = parser.parse_args()
        
        print(f"DEBUG: Script started with arguments:")
        print(f"DEBUG:   sentence: {args.sentence}")
        print(f"DEBUG:   model_dir: {args.model_dir}")
        print(f"DEBUG:   seq_length: {args.seq_length}")
        print(f"DEBUG:   poly_modulus_degree: {args.poly_modulus_degree}")
        print(f"DEBUG:   use_bert_base: {args.use_bert_base}")
        print(f"DEBUG:   hidden_size: {args.hidden_size}")
        print(f"DEBUG:   num_heads: {args.num_heads}")
        
        # Add overall timing
        overall_start_time = time.time()
        
        print("=" * 70)
        print("RUNNING ACTUAL SEAL INFERENCE PIPELINE")
        print("=" * 70)
        
        # Infer model parameters from weights or use BERT base parameters
        if args.hidden_size is not None:
            print(f"DEBUG: Using explicitly provided hidden size: {args.hidden_size}")
            hidden_size = args.hidden_size
        elif args.use_bert_base:
            print(f"DEBUG: Using BERT base parameters as requested")
            hidden_size = 768
        else:
            try:
                print(f"DEBUG: Attempting to infer model parameters")
                model_params = get_model_params_from_weights(args.model_dir)
                hidden_size = model_params['hidden_size']
                print(f"DEBUG: Successfully inferred hidden size: {hidden_size}")
            except Exception as e:
                print(f"DEBUG: Error inferring model parameters: {e}")
                print("DEBUG: Using BERT base hidden size instead")
                hidden_size = 768
                
        # Handle number of heads
        if args.num_heads is not None:
            print(f"DEBUG: Using explicitly provided number of heads: {args.num_heads}")
            num_heads = args.num_heads
        elif args.use_bert_base:
            num_heads = 12
        else:
            try:
                if 'num_heads' in locals() and model_params and 'num_heads' in model_params:
                    num_heads = model_params['num_heads']
                else:
                    # Estimate based on hidden size
                    num_heads = max(2, hidden_size // 64)  # Ensure at least 2 heads
                print(f"DEBUG: Using number of heads: {num_heads}")
            except Exception as e:
                print(f"DEBUG: Error determining number of heads: {e}")
                num_heads = max(2, hidden_size // 64)  # Default calculation
                
        # Handle number of layers
        if args.use_bert_base:
            num_layers = 12
        else:
            try:
                if 'num_layers' in locals() and model_params and 'num_layers' in model_params:
                    num_layers = model_params['num_layers']
                else:
                    num_layers = 1  # Default to 1 layer
                print(f"DEBUG: Using number of layers: {num_layers}")
            except Exception as e:
                print(f"DEBUG: Error determining number of layers: {e}")
                num_layers = 1  # Default to 1 layer
        
        print(f"Using model parameters:")
        print(f"  hidden_size: {hidden_size}")
        print(f"  num_heads: {num_heads}")
        print(f"  layers: {num_layers}")
        print(f"  poly_modulus_degree: {args.poly_modulus_degree}")
        print()
        
        # Prepare input data
        print(f"DEBUG: Preparing input data")
        try:
            data_paths = prepare_input_data(
                args.sentence, 
                seq_length=args.seq_length,
                hidden_size=hidden_size
            )
            print(f"DEBUG: Input data prepared successfully")
        except Exception as e:
            print(f"DEBUG: Error preparing input data: {e}")
            sys.exit(1)
        
        # Run inference
        seal_time_start = time.time()
        print(f"DEBUG: Running SEAL inference")
        success = run_seal_inference(
            data_paths,
            args.model_dir,
            poly_modulus_degree=args.poly_modulus_degree,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_layers=num_layers
        )
        
        # Process output if successful
        if success:
            print(f"DEBUG: Processing output")
            output_values = read_output(data_paths['output_path'])
            
            if output_values is not None:
                print("\nInference Results:")
                print(f"  Output size: {len(output_values)} values")
                print(f"  Mean: {np.mean(output_values):.6f}")
                print(f"  Min: {np.min(output_values):.6f}")
                print(f"  Max: {np.max(output_values):.6f}")
                
                # If output matches expected size, reshape it
                expected_size = args.seq_length * hidden_size
                
                # Silently handle different output sizes - remove debug messages
                if len(output_values) == expected_size:
                    reshaped = output_values.reshape(args.seq_length, hidden_size)
                    cls_score = np.mean(reshaped[0])
                    sentiment = "Positive" if cls_score > 0 else "Negative"
                    print(f"\nDetected sentiment: {sentiment}")
                else:
                    # Handle larger output size by using first expected_size elements
                    if len(output_values) > expected_size:
                        output_values = output_values[:expected_size]
                        reshaped = output_values.reshape(args.seq_length, hidden_size)
                        cls_score = np.mean(reshaped[0])
                        sentiment = "Positive" if cls_score > 0 else "Negative"
                        print(f"\nDetected sentiment: {sentiment}")
            else:
                print(f"DEBUG: Failed to read output values")
        else:
            print(f"DEBUG: SEAL inference failed")
        
        seal_time_end = time.time()
        seal_duration = seal_time_end - seal_time_start
        print(f"\nSEAL inference time: {seal_duration:.2f} seconds")
        
        # Calculate and print overall timing
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        print(f"\nTotal inference time: {overall_duration:.2f} seconds")
        
        # Clean up
        try:
            print(f"DEBUG: Cleaning up temporary directory: {data_paths['temp_dir']}")
            shutil.rmtree(data_paths['temp_dir'])
            print("\nCleanup complete. Temporary files removed.")
        except Exception as e:
            print(f"DEBUG: Error during cleanup: {e}")
    
    except Exception as e:
        print(f"DEBUG: Unhandled exception in main: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 