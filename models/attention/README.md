# Attention Mechanism Implementations

This directory contains different implementations of attention mechanisms for transformer models.

## 1. Standard Attention

The standard scaled dot-product attention used in the original Transformer paper:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Where:
- Q: Query matrix
- K: Key matrix
- V: Value matrix
- d_k: Dimension of the key vectors

Implementation: `standard_attention.py`

## 2. Inhibitor Attention

An alternative attention mechanism that uses Manhattan distance instead of dot product and inhibition instead of softmax:

```
Z_ij = ∑_k (1/γ) |Q_ik - K_jk|
H_ik = ∑_j (V_jk - Z_ij)^+
```

Where:
- Z_ij: Attention scores using Manhattan distance
- H_ik: Output after applying inhibition
- γ: Learnable scaling parameter
- ()^+: ReLU function (only keeps positive values)

Implementation: `inhibitor_attention.py`

## 3. Quadratic Inhibitor Attention

A quadratic approximation of the inhibitor mechanism that's compatible with CKKS (homomorphic encryption scheme):

```
H_ik ≈ ∑_j (V_jk - (15/(16γ) ||Q_i - K_j||²_2 + 3d/(16γ)))_+
```

Where:
- ||Q_i - K_j||²_2: Squared L2 norm (Euclidean distance squared)
- d: Dimension of the attention head
- γ: Learnable scaling parameter
- ()_+: ReLU function

This quadratic form uses the L2 norm instead of Manhattan distance and adds a dimension-dependent term.

Implementation: `quadratic_inhibitor_attention.py`

## Usage

When initializing the BERT model, specify the attention mechanism type:

```python
# Standard attention
model = BertModel(config_path, attention_type="standard")

# Inhibitor attention
model = BertModel(config_path, attention_type="inhibitor")

# Quadratic inhibitor attention
model = BertModel(config_path, attention_type="quadratic_inhibitor")
``` 