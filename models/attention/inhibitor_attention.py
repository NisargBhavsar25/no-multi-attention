import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InhibitorAttention(nn.Module):
    """
    Inhibitor attention mechanism using Manhattan distance instead of dot product
    and inhibition instead of softmax as described in the research paper.
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
        
        # Manhattan distance scaling factor (gamma) - initialize with a better scaling factor
        # Initialize with sqrt(attention_head_size) for better numerical stability
        self.gamma = nn.Parameter(torch.ones(1) * math.sqrt(self.attention_head_size))
        
        # Inhibition shift factor (alpha)
        self.alpha = nn.Parameter(torch.zeros(1))
    
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
        
        # Calculate Manhattan distance attention scores: Z_ij = ∑_k (1/γ) |Q_ik - K_jk|
        # Compute pairwise absolute differences
        # [batch, num_heads, seq_len_q, dim] - [batch, num_heads, dim, seq_len_k]
        # = [batch, num_heads, seq_len_q, seq_len_k, dim]
        query_expanded = query_layer.unsqueeze(3)  # [B, N, S_q, 1, D]
        key_expanded = key_layer.unsqueeze(2)      # [B, N, 1, S_k, D]
        
        # |Q_ik - K_jk|
        abs_diff = torch.abs(query_expanded - key_expanded)  # [B, N, S_q, S_k, D]
        
        # Sum over the last dimension and scale by 1/gamma with proper scaling factor
        # Apply scaling factor based on dimensionality for better numerical stability
        z_scores = torch.sum(abs_diff, dim=-1) / (self.gamma * math.sqrt(self.attention_head_size))  # [B, N, S_q, S_k]
        
        # Apply shifted inhibition: Z' = (Z - α)^+
        # This helps with zero inhibition score
        z_prime = F.relu(z_scores - self.alpha)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask from [B, S] to [B, 1, 1, S] for broadcasting
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Apply mask: set masked positions to very large values so they'll be zeroed by ReLU later
            z_prime = z_prime + (1.0 - extended_attention_mask) * 1e10
        
        # Apply inhibition: H_ik = ∑_j (V_jk - Z_ij)^+
        # For each query position and head, we compute V_jk - Z_ij for all keys/values
        v_expanded = value_layer.unsqueeze(2)  # [B, N, 1, S_k, D]
        z_expanded = z_prime.unsqueeze(-1)     # [B, N, S_q, S_k, 1]
        
        # Compute inhibited values
        inhibited = F.relu(v_expanded - z_expanded)  # [B, N, S_q, S_k, D]
        
        # Sum over the sequence length dimension (key/values)
        context_layer = torch.sum(inhibited, dim=3)  # [B, N, S_q, D]
        
        # Apply dropout
        context_layer = self.dropout(context_layer)
        
        # Reshape back to original format
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Apply output projection
        output = self.output(context_layer)
        
        return output 