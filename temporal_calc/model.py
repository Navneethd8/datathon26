"""
Lightweight Transformer Model for Temporal Accessibility Forecasting.

Optimized for M1 Mac with 8GB RAM:
- Small model (~50K parameters)
- Encoder-only architecture
- Efficient attention implementation
"""
import math
import torch
import torch.nn as nn
from typing import Optional

from config import MODEL_CONFIG


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position information.
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AccessibilityTransformer(nn.Module):
    """
    Lightweight Transformer for accessibility score prediction.
    
    Architecture:
    - Input projection to d_model
    - Positional encoding
    - Transformer encoder layers
    - Global average pooling
    - Regression head
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 20,
        pred_len: int = 1
    ):
        """
        Args:
            input_dim: Number of input features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            pred_len: Prediction horizon
        """
        super().__init__()
        
        self.d_model = d_model
        self.pred_len = pred_len
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # [batch, seq, features]
            norm_first=True    # Pre-norm for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            [batch, pred_len] predictions
        """
        # Project input
        x = self.input_proj(x)  # [batch, seq, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # [batch, seq, d_model]
        
        # Global average pooling across sequence
        x = x.mean(dim=1)  # [batch, d_model]
        
        # Regression output
        out = self.head(x)  # [batch, pred_len]
        
        return out
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(input_dim: int, **kwargs) -> AccessibilityTransformer:
    """
    Create model with default or custom configuration.
    
    Args:
        input_dim: Number of input features
        **kwargs: Override config values
    """
    config = {**MODEL_CONFIG, **kwargs}
    
    model = AccessibilityTransformer(
        input_dim=input_dim,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    )
    
    print(f"Created model with {model.count_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test model
    batch_size = 16
    seq_len = 10
    input_dim = 35  # Approximate number of features
    
    model = create_model(input_dim)
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Memory footprint: ~{model.count_parameters() * 4 / 1024:.1f} KB (float32)")
