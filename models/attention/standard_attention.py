import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StandardAttention(nn.Module):
    """
    Standard self-attention mechanism with scaled dot-product attention
    as used in the original BERT/Transformer architecture.
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
    
    def transpose_for_scores(self, x):
        """Reshape from [B, S, D] to [B, N, S, D/N] where N is number of heads"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # Linear projections
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        # Calculate scaled dot-product attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask from [B, S] to [B, 1, 1, S] for broadcasting
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert mask values: 0 -> -10000.0, 1 -> 0.0
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_scores = attention_scores + extended_attention_mask
        
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to original format
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Apply output projection
        output = self.output(context_layer)
        
        return output 