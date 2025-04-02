import argparse
import subprocess
import os
import yaml
import json
import matplotlib.pyplot as plt
import pandas as pd
import time
import logging
from datetime import datetime

def setup_logging(experiment_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(experiment_dir, 'experiment.log')),
            logging.StreamHandler()
        ]
    )

def run_training(config_path, attention_type, experiment_dir, activation_type="gelu"):
    """Run training for a specific attention type and activation function."""
    cmd = [
        "python", "train.py",
        "--config", config_path,
        "--attention_type", attention_type,
        "--activation_type", activation_type
    ]
    
    if activation_type == "gelu":
        logging.info(f"Training model with {attention_type} attention...")
    else:
        logging.info(f"Training model with {attention_type} attention and {activation_type} activation...")
        
    start_time = time.time()
    subprocess.run(cmd, check=True)
    end_time = time.time()
    training_time = end_time - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")
    return training_time

def run_evaluation(config_path, model_path, attention_type, results_path, activation_type="gelu"):
    """Run evaluation for a specific attention type and activation function."""
    if activation_type == "gelu":
        logging.info(f"Evaluating model with {attention_type} attention...")
    else:
        logging.info(f"Evaluating model with {attention_type} attention and {activation_type} activation...")
        
    cmd = [
        "python", "evaluate.py",
        "--model_path", model_path,
        "--config", config_path,
        "--attention_type", attention_type,
        "--activation_type", activation_type,
        "--output_file", results_path
    ]
    subprocess.run(cmd, check=True)
    
    # Load and return the results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Add activation type to results for display
    if activation_type != "gelu":
        results["model_type"] = f"{attention_type} ({activation_type})"
    else:
        results["model_type"] = attention_type
        
    return results

def create_comparison_plots(results, experiment_dir):
    """Create plots comparing metrics across model types."""
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Use model_type instead of attention_type for display if it exists
    x_column = "model_type" if "model_type" in df.columns else "attention_type"
    
    # Accuracy comparison
    plt.figure(figsize=(12, 8))
    ax = df.plot(x=x_column, y='accuracy', kind='bar', color='skyblue')
    plt.title('Accuracy Comparison', fontsize=16)
    plt.xlabel('Model Type', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    for i, v in enumerate(df['accuracy']):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=12)
    plt.savefig(os.path.join(experiment_dir, 'accuracy_comparison.png'), dpi=300)
    
    # Latency comparison
    plt.figure(figsize=(12, 8))
    ax = df.plot(x=x_column, y='avg_latency', kind='bar', color='lightgreen')
    plt.title('Average Inference Latency Comparison', fontsize=16)
    plt.xlabel('Model Type', fontsize=14)
    plt.ylabel('Latency (seconds)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    for i, v in enumerate(df['avg_latency']):
        ax.text(i, v + 0.0005, f"{v:.4f}", ha='center', fontsize=12)
    plt.savefig(os.path.join(experiment_dir, 'latency_comparison.png'), dpi=300)
    
    # Memory usage comparison
    plt.figure(figsize=(12, 8))
    ax = df.plot(x=x_column, y='avg_memory_usage', kind='bar', color='salmon')
    plt.title('Average Memory Usage Comparison', fontsize=16)
    plt.xlabel('Model Type', fontsize=14)
    plt.ylabel('Memory Usage (MB)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    for i, v in enumerate(df['avg_memory_usage']):
        ax.text(i, v + 1, f"{v:.2f}", ha='center', fontsize=12)
    plt.savefig(os.path.join(experiment_dir, 'memory_comparison.png'), dpi=300)
    
    # Training time comparison
    plt.figure(figsize=(12, 8))
    ax = df.plot(x=x_column, y='training_time', kind='bar', color='mediumpurple')
    plt.title('Training Time Comparison', fontsize=16)
    plt.xlabel('Model Type', fontsize=14)
    plt.ylabel('Training Time (seconds)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    for i, v in enumerate(df['training_time']):
        ax.text(i, v + 50, f"{v:.2f}", ha='center', fontsize=12)
    plt.savefig(os.path.join(experiment_dir, 'training_time_comparison.png'), dpi=300)
    
    # Create a summary table as CSV
    summary_file = os.path.join(experiment_dir, 'results_summary.csv')
    df.to_csv(summary_file, index=False)
    logging.info(f"Results summary saved to {summary_file}")

def run_experiment(config_path):
    """Run the full experiment comparing attention mechanisms."""
    # Load experiment configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config['experiment']['name']
    experiment_dir = os.path.join(config['logging']['log_dir'], f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(experiment_dir)
    
    logging.info(f"Starting full experiment: {config['experiment']['description']}")
    logging.info(f"Using configuration from: {config_path}")
    
    # Save a copy of the configuration
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Results collection
    all_results = []
    
    # Check configuration options for special combinations
    run_standard_relu = config['experiment'].get('use_relu_activation', False)
    run_quadratic_inhibitor_relu = config['experiment'].get('use_quadratic_inhibitor_relu', False)
    
    # Run experiment for each attention type
    for attention_type in config['experiment']['attention_types']:
        logging.info(f"Testing attention type: {attention_type}")
        
        # Run training with default GELU activation
        training_time = run_training(config_path, attention_type, experiment_dir)
        
        # Path to the trained model
        model_path = os.path.join(config['logging']['log_dir'], f"{attention_type}_attention", "final_model")
        
        # Results path
        results_path = os.path.join(experiment_dir, f"results_{attention_type}.json")
        
        # Run evaluation
        eval_results = run_evaluation(config_path, model_path, attention_type, results_path)
        
        # Add attention type and training time to results
        eval_results['attention_type'] = attention_type
        eval_results['training_time'] = training_time
        
        # Collect results
        all_results.append(eval_results)
    
    # Run the standard attention with ReLU if specified
    if run_standard_relu:
        attention_type = "standard"
        activation_type = "relu"
        logging.info(f"Testing standard attention with ReLU activation")
        
        # Run training with ReLU activation
        training_time = run_training(config_path, attention_type, experiment_dir, activation_type)
        
        # Path to the trained model
        model_path = os.path.join(config['logging']['log_dir'], f"{attention_type}_attention_{activation_type}_activation", "final_model")
        
        # Results path
        results_path = os.path.join(experiment_dir, f"results_{attention_type}_{activation_type}.json")
        
        # Run evaluation
        eval_results = run_evaluation(config_path, model_path, attention_type, results_path, activation_type)
        
        # Add attention type, activation type and training time to results
        eval_results['attention_type'] = attention_type
        eval_results['activation_type'] = activation_type
        eval_results['training_time'] = training_time
        
        # Collect results
        all_results.append(eval_results)
    
    # Run the quadratic inhibitor attention with ReLU if specified
    if run_quadratic_inhibitor_relu:
        attention_type = "quadratic_inhibitor"
        activation_type = "relu"
        logging.info(f"Testing quadratic inhibitor attention with ReLU activation")
        
        # Run training with ReLU activation
        training_time = run_training(config_path, attention_type, experiment_dir, activation_type)
        
        # Path to the trained model
        model_path = os.path.join(config['logging']['log_dir'], f"{attention_type}_attention_{activation_type}_activation", "final_model")
        
        # Results path
        results_path = os.path.join(experiment_dir, f"results_{attention_type}_{activation_type}.json")
        
        # Run evaluation
        eval_results = run_evaluation(config_path, model_path, attention_type, results_path, activation_type)
        
        # Add attention type, activation type and training time to results
        eval_results['attention_type'] = attention_type
        eval_results['activation_type'] = activation_type
        eval_results['training_time'] = training_time
        
        # Collect results
        all_results.append(eval_results)
    
    # Save all results
    with open(os.path.join(experiment_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparison plots
    create_comparison_plots(all_results, experiment_dir)
    
    logging.info(f"Experiment completed. Results saved to {experiment_dir}")
    
    return experiment_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/attention_comparison_config.yaml',
                      help='Path to experiment config file')
    args = parser.parse_args()
    
    run_experiment(args.config)

if __name__ == '__main__':
    main() 