import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import yaml
import os

class BaseTransformerModel(nn.Module):
    def __init__(self, config_path: str):
        super().__init__()
        self.config = self._load_config(config_path)
        self.model_config = self.config['model']
        
        # Initialize basic components
        self.embeddings = None  # To be implemented by child classes
        self.encoder = None    # To be implemented by child classes
        self.classifier = None # To be implemented by child classes
        
        # Initialize metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_accuracy': [],
            'latency': [],
            'memory_usage': []
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        To be implemented by child classes with specific architectures.
        """
        raise NotImplementedError
    
    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss for training."""
        loss_fct = nn.CrossEntropyLoss()
        # Check if labels need reshaping
        if labels.dim() == 1 and outputs.dim() > 1:
            return loss_fct(outputs, labels)
        else:
            return loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))
    
    def measure_latency(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> float:
        """Measure inference latency."""
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            _ = self(input_ids, attention_mask)
            end_time.record()
            
            torch.cuda.synchronize()
            return start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
    
    def measure_memory_usage(self) -> float:
        """Measure model memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0
    
    def save_model(self, path: str):
        """Save model and configuration."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, 'model.pt'))
        with open(os.path.join(path, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
    
    def load_model(self, path: str):
        """Load model from saved checkpoint."""
        self.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
    
    def get_metrics(self) -> Dict[str, list]:
        """Return collected metrics."""
        return self.metrics 