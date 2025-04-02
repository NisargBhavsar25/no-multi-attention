# Attention Mechanism Comparison Experiment

This experiment compares the performance of different attention mechanisms in BERT-like transformer models on the IMDB sentiment classification task.

## Attention Mechanisms

### 1. Standard Attention

The standard attention mechanism used in the original BERT/Transformer papers, which uses scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

### 2. Inhibitor Attention

An alternative attention mechanism that replaces the dot product with Manhattan distance and softmax with an inhibition mechanism:

```
Z_ij = ∑_k (1/γ) |Q_ik - K_jk|
H_ik = ∑_j (V_jk - Z_ij)^+
```

Where:
- Z_ij: Attention scores based on Manhattan distance
- H_ik: Output after applying inhibition (ReLU)
- γ: Scaling parameter
- ^+: ReLU function

## Running the Experiment

To run the experiment:

```bash
python run_experiment.py --config experiments/attention_comparison_config.yaml
```

## Results

The experiment will produce:

1. Models trained with each attention mechanism
2. Performance metrics:
   - Accuracy on IMDB test set
   - Inference latency
   - Memory usage
   - Training time
3. Comparison visualizations:
   - Bar charts comparing the metrics
   - Results summary in JSON format

## Configuration

You can modify `attention_comparison_config.yaml` to:
- Change model hyperparameters
- Adjust training settings
- Configure evaluation metrics
- Add more attention mechanisms

## Output Location

Results are stored in `experiments/results/attention_comparison_{timestamp}/` 