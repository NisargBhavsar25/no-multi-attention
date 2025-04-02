import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from models.base_model import BaseTransformerModel
from models.attention.standard_attention import StandardAttention
from models.attention.inhibitor_attention import InhibitorAttention
from models.attention.quadratic_inhibitor_attention import QuadraticInhibitorAttention

class BertEmbeddings(nn.Module):
    """
    BERT embeddings consisting of token, position, and token type embeddings.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.word_embeddings = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.position_embeddings = nn.Embedding(config['max_position_embeddings'], config['hidden_size'])
        self.token_type_embeddings = nn.Embedding(config['type_vocab_size'], config['hidden_size'])
        
        self.layer_norm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        
        # Create position ids tensor once
        self.register_buffer(
            "position_ids", 
            torch.arange(config['max_position_embeddings']).expand((1, -1))
        )
    
    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertLayer(nn.Module):
    """
    BERT layer consisting of attention and feed-forward networks.
    """
    def __init__(self, config: Dict[str, Any], attention_type: str = "standard"):
        super().__init__()
        
        # Initialize attention based on specified type
        if attention_type == "standard":
            self.attention = StandardAttention(config)
        elif attention_type == "inhibitor":
            self.attention = InhibitorAttention(config)
        elif attention_type == "quadratic_inhibitor":
            self.attention = QuadraticInhibitorAttention(config)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
        
        # Add attention output LayerNorm
        self.attention_layernorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.attention_dropout = nn.Dropout(config['hidden_dropout_prob'])
        
        # Feed-forward network
        self.intermediate = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.intermediate_act_fn = nn.GELU()
        self.output = nn.Linear(config['intermediate_size'], config['hidden_size'])
        
        # Output LayerNorm
        self.output_layernorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.output_dropout = nn.Dropout(config['hidden_dropout_prob'])
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection and layer norm
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layernorm(hidden_states + attention_output)
        
        # Feed-forward network with residual connection and layer norm
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layernorm(attention_output + layer_output)
        
        return layer_output

class BertEncoder(nn.Module):
    """
    Stack of BERT layers.
    """
    def __init__(self, config: Dict[str, Any], attention_type: str = "standard"):
        super().__init__()
        self.layers = nn.ModuleList([
            BertLayer(config, attention_type) for _ in range(config['num_hidden_layers'])
        ])
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class BertPooler(nn.Module):
    """
    Pool the output of the last layer for classification tasks.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Use the [CLS] token representation (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertClassifier(nn.Module):
    """
    Classification head for BERT.
    """
    def __init__(self, config: Dict[str, Any], num_labels: int = 2):
        super().__init__()
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.classifier = nn.Linear(config['hidden_size'], num_labels)
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class BertModel(BaseTransformerModel):
    """
    BERT model with configurable attention mechanism.
    """
    def __init__(self, config_path: str, attention_type: str = "standard"):
        super().__init__(config_path)
        
        # Initialize embeddings
        self.embeddings = BertEmbeddings(self.model_config)
        
        # Initialize encoder with specified attention type
        self.encoder = BertEncoder(self.model_config, attention_type)
        
        # Initialize pooler and classifier
        self.pooler = BertPooler(self.model_config)
        self.classifier = BertClassifier(self.model_config)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Generate embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        # Encode sequence
        sequence_output = self.encoder(embedding_output, attention_mask)
        
        # Pool and classify
        pooled_output = self.pooler(sequence_output)
        logits = self.classifier(pooled_output)
        
        return logits 