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

### 3. Quadratic Inhibitor Attention

An extension of the Inhibitor Attention that uses quadratic distance instead of Manhattan distance:

```
Z_ij = ∑_k (1/γ) (Q_ik - K_jk)²
H_ik = ∑_j (V_jk - Z_ij)^+
```

Where:
- Z_ij: Attention scores based on quadratic distance
- H_ik: Output after applying inhibition (ReLU)
- γ: Scaling parameter
- ^+: ReLU function

### 4. ConsMax Attention

An alternative attention mechanism that replaces the softmax function with ConsMax activation:

```
ConSmax(S_i) = (e^(S_i-β))/γ = C × e^S_i, where C = -e^β/γ
```

Where:
- S_i: Attention score for position i
- β: Shift parameter (learnable)
- γ: Scale parameter (learnable)
- C: Constant factor during inference

Unlike softmax, ConsMax:
- Does not normalize the probability vector to sum to 1
- Includes learnable parameters β and γ
- Maintains β and γ as independent parameters during training to mitigate overflow
- Can merge parameters into a single constant during inference

### 5. ApproxExp Attention

An extension of ConsMax that replaces the exponential function with an algebraic approximation:

```
ApproxExp(S_i) = ((1 + (S_i-β)/(2^r))^(2^r))/γ, x ≤ 0, with r = 7
```

Where:
- S_i: Attention score for position i
- β: Shift parameter (learnable)
- γ: Scale parameter (learnable)
- r: Fixed parameter (r=7)

The exponential approximation used is:
```
EXP(x) ≈ (1 + x/2^r)^(2^r), x ≤ 0
```

Benefits of this approach:
- Avoids expensive exponential operations
- May improve computational efficiency
- Shares the same overall structure as ConsMax but with a different activation function
- Still includes learnable parameters β and γ

### 6. Standard Attention with ReLU Activation

A variation of the standard attention model where all GELU activations in the feed-forward networks are replaced with ReLU activations:

```
ReLU(x) = max(0, x)
```

While the attention mechanism itself remains unchanged (using standard attention with softmax), the GELU activation in each transformer block's feed-forward network is replaced with ReLU.

This allows us to study the impact of the activation function choice separate from the attention mechanism choice.

### 7. Quadratic Inhibitor Attention with ReLU Activation

A variation that combines the quadratic inhibitor attention mechanism with ReLU activations in the feed-forward networks:

```
Z_ij = ∑_k (1/γ) (Q_ik - K_jk)²
H_ik = ∑_j (V_jk - Z_ij)^+
```

And in the feed-forward networks:
```
ReLU(x) = max(0, x)
```

This combination allows us to study the interaction effects between attention mechanism choice and activation function choice, particularly for distance-based attention mechanisms like quadratic inhibitor that already use ReLU-like operations internally.

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
- Control dataset size:
  - `num_train_samples`: Limit the number of training samples (-1 for all samples)
  - `num_test_samples`: Limit the number of test samples (-1 for all samples)

## Output Location

Results are stored in `experiments/results/attention_comparison_{timestamp}/` 