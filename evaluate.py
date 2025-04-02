import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import yaml
import os
import json
from models.base_model import BaseTransformerModel
from models.bert_model import BertModel
import logging
from tqdm import tqdm
import numpy as np

def setup_logging(config):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config['logging']['log_dir'], 'evaluation.log')),
            logging.StreamHandler()
        ]
    )

def load_test_data(config):
    # Load IMDB dataset
    dataset = load_dataset("imdb", split="test")
    
    # Check if we should limit the number of test samples
    num_test_samples = config['dataset'].get('num_test_samples', -1)
    if num_test_samples > 0 and num_test_samples < len(dataset):
        dataset = dataset.select(range(num_test_samples))
        logging.info(f"Limiting to {num_test_samples} test samples as specified in config")
    
    # Log the dataset size
    logging.info(f"Evaluating on test dataset: {len(dataset)} test samples")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create data collator for batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding=False,  # We'll use the data collator for padding
            truncation=True,
            max_length=config['training']['max_seq_length'],
            return_tensors=None  # Don't convert to tensors yet
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Add labels
    tokenized_dataset = tokenized_dataset.add_column('labels', dataset['label'])
    
    # Create data loader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config['evaluation']['batch_size'],
        collate_fn=data_collator
    )
    
    return dataloader

def evaluate_model(model, dataloader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    metrics = {
        'accuracy': 0.0,
        'latency': [],
        'memory_usage': [],
        'predictions': []
    }
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Measure latency
            latency = model.measure_latency(input_ids, attention_mask)
            metrics['latency'].append(latency)
            
            # Measure memory usage
            memory = model.measure_memory_usage()
            metrics['memory_usage'].append(memory)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=-1)
            
            # Calculate accuracy
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Store predictions
            metrics['predictions'].extend(predictions.cpu().numpy().tolist())
    
    # Calculate final metrics
    metrics['accuracy'] = correct / total
    metrics['avg_latency'] = np.mean(metrics['latency'])
    metrics['std_latency'] = np.std(metrics['latency'])
    metrics['avg_memory_usage'] = np.mean(metrics['memory_usage'])
    
    return metrics

def save_results(metrics, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--attention_type', type=str, default='standard', 
                        choices=['standard', 'inhibitor', 'quadratic_inhibitor', 'consmax', 'approx_exp'], 
                        help='Type of attention mechanism used in the model')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save evaluation results')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(config)
    
    # Log evaluation details
    logging.info(f"Evaluating model with {args.attention_type} attention mechanism")
    
    # Load model with the right attention type
    model = BertModel(args.config, attention_type=args.attention_type)
    model.load_model(args.model_path)
    
    # Load test data
    test_dataloader = load_test_data(config)
    
    # Evaluate model
    metrics = evaluate_model(model, test_dataloader, config)
    
    # Log results
    logging.info(f"Attention Type: {args.attention_type}")
    logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"Average Latency: {metrics['avg_latency']:.4f} seconds")
    logging.info(f"Latency Std: {metrics['std_latency']:.4f} seconds")
    logging.info(f"Average Memory Usage: {metrics['avg_memory_usage']:.2f} MB")
    
    # Save results
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(config['logging']['log_dir'], f'evaluation_results_{args.attention_type}.json')
    
    save_results(metrics, output_path)
    logging.info(f"Results saved to {output_path}")

if __name__ == '__main__':
    main() 