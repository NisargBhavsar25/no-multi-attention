import torch
import struct
import os
import numpy as np

def convert_weights(model_path, output_dir):
    # Load your trained model
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of layers from the model structure
    # Find keys that match 'encoder.layers.{n}.attention.query.weight' pattern
    layer_keys = [k for k in model_state_dict.keys() if '.attention.query.weight' in k]
    num_layers = len(layer_keys)
    
    print(f"Found {num_layers} layers in the model")
    
    # For each weight type, create a binary file
    for weight_type in ['wq', 'wk', 'wv', 'wo', 'ff1', 'ff2']:
        with open(os.path.join(output_dir, f"{weight_type}.bin"), 'wb') as f:
            # Write number of layers (uint32)
            f.write(struct.pack('I', num_layers))
            
            # For each layer, write its weights
            for layer_idx in range(num_layers):
                # Get weight tensor for current layer and type
                if weight_type == 'wq':
                    key = f'encoder.layers.{layer_idx}.attention.query.weight'
                elif weight_type == 'wk':
                    key = f'encoder.layers.{layer_idx}.attention.key.weight'
                elif weight_type == 'wv':
                    key = f'encoder.layers.{layer_idx}.attention.value.weight'
                elif weight_type == 'wo':
                    key = f'encoder.layers.{layer_idx}.attention.output.weight'
                elif weight_type == 'ff1':
                    key = f'encoder.layers.{layer_idx}.intermediate.weight'
                elif weight_type == 'ff2':
                    key = f'encoder.layers.{layer_idx}.output.weight'
                
                # Check if key exists in the model
                if key not in model_state_dict:
                    print(f"Warning: Key {key} not found in model state dict")
                    continue
                
                # Get the weights
                weights = model_state_dict[key].detach().cpu().numpy()
                
                # Flatten the weights
                weights_flat = weights.flatten()
                
                # Write number of elements (uint32)
                f.write(struct.pack('I', len(weights_flat)))
                
                # Write weight values (doubles)
                for w in weights_flat:
                    f.write(struct.pack('d', float(w)))
    
    print(f"Conversion complete. Weight files saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert PyTorch model weights to binary format')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the PyTorch model file (.pt or .pth)')
    parser.add_argument('--output_dir', type=str, default='./model', help='Directory to save the converted weights')
    args = parser.parse_args()
    
    convert_weights(args.model_path, args.output_dir) 