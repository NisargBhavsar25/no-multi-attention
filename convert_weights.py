import torch
import struct
import os
import numpy as np

def convert_weights(model_path, output_dir):
    # Load your trained model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract weights for each layer
    num_layers = len(model['layers']) if 'layers' in model else 1
    
    # For each weight type, create a binary file
    for weight_type in ['wq', 'wk', 'wv', 'wo', 'ff1', 'ff2']:
        with open(os.path.join(output_dir, f"{weight_type}.bin"), 'wb') as f:
            # Write number of layers (uint32)
            f.write(struct.pack('I', num_layers))
            
            # For each layer, write its weights
            for layer_idx in range(num_layers):
                # Get weight tensor for current layer and type
                if 'layers' in model:
                    # Multi-layer model
                    if weight_type == 'wq':
                        weights = model['layers'][layer_idx]['attention']['query'].weight.detach().cpu().numpy()
                    elif weight_type == 'wk':
                        weights = model['layers'][layer_idx]['attention']['key'].weight.detach().cpu().numpy()
                    elif weight_type == 'wv':
                        weights = model['layers'][layer_idx]['attention']['value'].weight.detach().cpu().numpy()
                    elif weight_type == 'wo':
                        weights = model['layers'][layer_idx]['attention']['output'].weight.detach().cpu().numpy()
                    elif weight_type == 'ff1':
                        weights = model['layers'][layer_idx]['intermediate'].weight.detach().cpu().numpy()
                    elif weight_type == 'ff2':
                        weights = model['layers'][layer_idx]['output'].weight.detach().cpu().numpy()
                else:
                    # Single layer model
                    if weight_type == 'wq':
                        weights = model['attention']['query'].weight.detach().cpu().numpy()
                    elif weight_type == 'wk':
                        weights = model['attention']['key'].weight.detach().cpu().numpy()
                    elif weight_type == 'wv':
                        weights = model['attention']['value'].weight.detach().cpu().numpy()
                    elif weight_type == 'wo':
                        weights = model['attention']['output'].weight.detach().cpu().numpy() 
                    elif weight_type == 'ff1':
                        weights = model['intermediate'].weight.detach().cpu().numpy()
                    elif weight_type == 'ff2':
                        weights = model['output'].weight.detach().cpu().numpy()
                
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