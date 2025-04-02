import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm
import yaml
import os
from models.base_model import BaseTransformerModel
from models.bert_model import BertModel
import logging

def setup_logging(config):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config['logging']['log_dir'], 'training.log')),
            logging.StreamHandler()
        ]
    )

def load_data(config):
    # Load IMDB dataset
    dataset = load_dataset("imdb")
    
    # Log the dataset size
    logging.info(f"Training on full dataset: {len(dataset['train'])} training samples, {len(dataset['test'])} test samples")
    
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
    
    # Tokenize datasets
    tokenized_train = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    tokenized_test = dataset['test'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['test'].column_names
    )
    
    # Add labels
    tokenized_train = tokenized_train.add_column('labels', dataset['train']['label'])
    tokenized_test = tokenized_test.add_column('labels', dataset['test']['label'])
    
    # Create data loaders with collation function
    train_dataloader = DataLoader(
        tokenized_train,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=data_collator
    )
    
    eval_dataloader = DataLoader(
        tokenized_test,
        batch_size=config['evaluation']['batch_size'],
        collate_fn=data_collator
    )
    
    return train_dataloader, eval_dataloader

def train(model, train_dataloader, eval_dataloader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Ensure learning rate is a float
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    num_training_steps = len(train_dataloader) * config['training']['num_train_epochs']
    
    for epoch in range(config['training']['num_train_epochs']):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = model.compute_loss(outputs, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
            
            # Measure and log metrics
            if progress_bar.n % config['logging']['eval_steps'] == 0 and progress_bar.n > 0:
                model.metrics['train_loss'].append(total_loss / (progress_bar.n + 1))
                model.metrics['latency'].append(model.measure_latency(input_ids, attention_mask))
                model.metrics['memory_usage'].append(model.measure_memory_usage())
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        model.metrics['val_accuracy'].append(accuracy)
        logging.info(f'Epoch {epoch + 1} - Validation Accuracy: {accuracy:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_steps'] == 0 or (epoch + 1) == config['training']['num_train_epochs']:
            model.save_model(os.path.join(config['logging']['log_dir'], f'checkpoint-{epoch + 1}'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--attention_type', type=str, default='standard', 
                        choices=['standard', 'inhibitor', 'quadratic_inhibitor'], 
                        help='Type of attention mechanism to use')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create log directory with attention type info
    log_dir = os.path.join(config['logging']['log_dir'], f"{args.attention_type}_attention")
    config['logging']['log_dir'] = log_dir
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(config)
    
    # Log experiment details
    logging.info(f"Starting training with {args.attention_type} attention mechanism")
    
    # Load data
    train_dataloader, eval_dataloader = load_data(config)
    
    # Initialize model with specified attention type
    model = BertModel(args.config, attention_type=args.attention_type)
    
    # Train model
    train(model, train_dataloader, eval_dataloader, config)
    
    # Save final model
    model.save_model(os.path.join(log_dir, 'final_model'))
    logging.info(f"Training completed. Model saved to {os.path.join(log_dir, 'final_model')}")

if __name__ == '__main__':
    main() 