import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QuadraticInhibitorAttention(nn.Module):
    """
    Quadratic Inhibitor attention mechanism using the quadratic approximation formula:
    H_ik ≈ ∑_j (V_jk - (15/(16γ) ||Q_i - K_j||²_2 + 3d/(16γ)))_+
    
    This is a quadratic form approximation of the inhibitor mechanism compatible with CKKS.
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear layers for Q, K, V projections
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(self.all_head_size, self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config['attention_probs_dropout_prob'])
        
        # Gamma parameter for scaling the quadratic form
        # Initialize with sqrt(attention_head_size) for better numerical stability
        self.gamma = nn.Parameter(torch.ones(1) * math.sqrt(self.attention_head_size))
        
        # Constant term derived from head dimension (3d/16)
        self.register_buffer('dim_scale', torch.tensor(3.0 * self.attention_head_size / 16.0))
    
    def transpose_for_scores(self, x):
        """Reshape from [B, S, D] to [B, N, S, D/N] where N is number of heads"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def normalize_attention(self, attention_probs):
        """Add row-wise normalization to prevent attention collapse"""
        # Add small epsilon to avoid division by zero
        row_sum = attention_probs.sum(dim=-1, keepdim=True) + 1e-6
        return attention_probs / row_sum
    
    def forward(self, hidden_states, attention_mask=None):
        # Linear projections
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        # Calculate quadratic form for inhibition scores
        # [batch, num_heads, seq_len_q, 1, dim]
        query_expanded = query_layer.unsqueeze(3)
        # [batch, num_heads, 1, seq_len_k, dim]
        key_expanded = key_layer.unsqueeze(2)
        
        # Calculate squared L2 norm: ||Q_i - K_j||²_2
        # First compute difference: (Q_i - K_j)
        diff = query_expanded - key_expanded  # [B, N, S_q, S_k, D]
        
        # Square and sum over the last dimension to get squared L2 norm
        # [B, N, S_q, S_k]
        squared_l2 = torch.sum(diff * diff, dim=-1)
        
        # Calculate Z_ij = (15/(16γ) * ||Q_i - K_j||²_2 + 3d/(16γ))
        # Apply proper scaling including the dimension factor
        coef = 15.0 / (16.0 * self.gamma * math.sqrt(self.attention_head_size))
        
        # Compute Z_ij with scaled coefficient
        z_scores = coef * squared_l2 + self.dim_scale / (self.gamma * math.sqrt(self.attention_head_size))  # [B, N, S_q, S_k]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask from [B, S] to [B, 1, 1, S] for broadcasting
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Apply mask: set masked positions to very large values so they'll be zeroed by ReLU later
            z_scores = z_scores + (1.0 - extended_attention_mask) * 1e10
        
        # Apply inhibition: H_ik = ∑_j (V_jk - Z_ij)^+
        # [B, N, 1, S_k, D]
        v_expanded = value_layer.unsqueeze(2)
        # [B, N, S_q, S_k, 1]
        z_expanded = z_scores.unsqueeze(-1)
        
        # Compute inhibited values: (V_jk - Z_ij)^+
        # [B, N, S_q, S_k, D]
        inhibited = F.relu(v_expanded - z_expanded)
        
        # Sum over the sequence length dimension (key/values)
        # [B, N, S_q, D]
        context_layer = torch.sum(inhibited, dim=3)
        
        # Apply dropout
        context_layer = self.dropout(context_layer)
        
        # Reshape back to original format
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Apply output projection
        output = self.output(context_layer)
        
        return output 