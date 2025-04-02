# BERT Attention Mechanism Comparison - Full Experiment

This document provides instructions for running the full experiment comparing different attention mechanisms in BERT-like transformer models on the IMDB sentiment classification task.

## Attention Mechanisms

The experiment compares three different attention mechanisms:

1. **Standard Attention** - The original scaled dot-product attention
2. **Inhibitor Attention** - Manhattan distance-based attention with ReLU inhibition
3. **Quadratic Inhibitor Attention** - Quadratic approximation compatible with CKKS

## Running the Full Experiment

The full experiment will train and evaluate each model on the complete IMDB dataset using the full model configuration. This will take several hours to complete, depending on your hardware.

To run the full experiment:

```bash
python run_experiment.py --config experiments/attention_comparison_config.yaml
```

## Hardware Requirements

For the full experiment, we recommend:
- GPU with at least 8GB VRAM
- At least 16GB RAM
- About 20GB free disk space for models and datasets

## Configuration Details

The full experiment uses the following configuration:
- Model size: 768 hidden dimensions
- 6 transformer layers
- 12 attention heads
- Sequence length: 256 tokens
- 3 training epochs
- Batch size: 16
- Complete IMDB dataset (25,000 training samples, 25,000 test samples)

## Results

After completion, results will be available in:
```
experiments/results/attention_comparison_full_{timestamp}/
```

The results include:
- Accuracy comparison
- Latency measurements
- Memory usage statistics 
- Training time comparison
- Raw data in JSON format
- Performance visualizations
- CSV summary of all metrics

## Running Individual Models

If you want to train or evaluate a specific attention mechanism separately:

```bash
# Training
python train.py --config experiments/attention_comparison_config.yaml --attention_type standard
python train.py --config experiments/attention_comparison_config.yaml --attention_type inhibitor  
python train.py --config experiments/attention_comparison_config.yaml --attention_type quadratic_inhibitor

# Evaluation
python evaluate.py --model_path experiments/results/attention_comparison_full/standard_attention/final_model --config experiments/attention_comparison_config.yaml --attention_type standard
```

## Extending the Experiment

To add a new attention mechanism:
1. Implement the new mechanism in `models/attention/`
2. Update `models/bert_model.py` to include the new mechanism
3. Add the new mechanism type to the config file's `attention_types` list
4. Update the command-line argument choices in train.py and evaluate.py 