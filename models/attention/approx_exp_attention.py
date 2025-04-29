import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ApproxExpAttention(nn.Module):
    """
    Attention mechanism using an algebraic approximation of exponential function
    instead of the standard exp() used in ConsMax.
    
    The approximation used is:
    EXP(x) ≈ (1 + x/2r)^(2r), x ≤ 0, with r = 7
    
    This is then used in a ConsMax-like formula:
    ApproxExp(S_i) = (approx_exp(S_i-β))/γ
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
        
        # ConsMax parameters - initialize with better values for stability
        self.beta = nn.Parameter(torch.zeros(1))  # Shift parameter
        # Scale gamma by the dimension for better numerical stability
        self.gamma = nn.Parameter(torch.ones(1) * math.sqrt(self.attention_head_size))  
        
        # Approximation parameter r=7 (fixed)
        self.r = 7
    
    def transpose_for_scores(self, x):
        """Reshape from [B, S, D] to [B, N, S, D/N] where N is number of heads"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def approx_exp(self, x):
        """
        Algebraic approximation of exponential function
        EXP(x) ≈ (1 + x/2r)^(2r), x ≤ 0, with r = 7
        
        For numerical stability, we clamp x to be <= 0
        """
        # Ensure x is non-positive for approximation validity
        x = torch.clamp(x, max=0)
        
        # Apply the approximation formula correctly
        # 2r = 14, not 2^r = 128
        denominator = 2 * self.r
        base = 1 + x / denominator
        result = torch.pow(base, denominator)
        
        return result
    
    def apply_approx_exp_attention(self, scores, mask=None):
        """
        Apply approximated exponential function to attention scores
        ApproxExp(S_i) = (approx_exp(S_i-β))/γ
        """
        # Apply mask if provided
        if mask is not None:
            scores = scores + (1.0 - mask) * -10000.0
        
        # Apply the ApproxExp formula: (approx_exp(S_i-β))/γ
        scores_shifted = scores - self.beta
        
        # Similar to ConsMax, we apply the same max-normalization trick for numerical stability
        scores_max, _ = torch.max(scores_shifted, dim=-1, keepdim=True)
        scores_shifted = scores_shifted - scores_max
        
        # Apply our approximated exponential function and scale by gamma
        attention_probs = self.approx_exp(scores_shifted) / (self.gamma * math.sqrt(self.attention_head_size))
        
        # Row-wise normalization to ensure attention probabilities sum to 1
        attention_probs_sum = attention_probs.sum(dim=-1, keepdim=True) + 1e-6
        attention_probs = attention_probs / attention_probs_sum
        
        # Apply dropout
        attention_probs = self.dropout(attention_probs)
        
        return attention_probs
    
    def forward(self, hidden_states, attention_mask=None):
        # Linear projections
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        # Calculate dot product attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply ApproxExp attention
        if attention_mask is not None:
            # Convert mask from [B, S] to [B, 1, 1, S] for broadcasting
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            extended_attention_mask = None
            
        attention_probs = self.apply_approx_exp_attention(attention_scores, extended_attention_mask)
        
        # Apply attention probabilities to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to original format
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Apply output projection
        output = self.output(context_layer)
        
        return output 