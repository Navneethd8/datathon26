"""
GNN Model Architectures
Implements GCN, GAT, and GraphSAGE models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from typing import Optional


class GATModel(nn.Module):
    """Graph Attention Network"""
    
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int,
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.2):
        """
        Initialize GAT model
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            embedding_dim: Output embedding dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # First layer
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                     heads=num_heads, dropout=dropout))
        
        # Final layer (single head for embedding)
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * num_heads, embedding_dim, 
                                     heads=1, dropout=dropout))
        else:
            self.convs.append(GATConv(input_dim, embedding_dim, heads=1, dropout=dropout))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Optional edge attributes [E, edge_dim]
            
        Returns:
            Node embeddings [N, embedding_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x


class GCNModel(nn.Module):
    """Graph Convolutional Network"""
    
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int,
                 num_layers: int = 3, dropout: float = 0.2):
        """
        Initialize GCN model
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            embedding_dim: Output embedding dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super(GCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Final layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, embedding_dim))
        else:
            self.convs.append(GCNConv(input_dim, embedding_dim))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Optional edge attributes (not used in GCN)
            
        Returns:
            Node embeddings [N, embedding_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x


class GraphSAGEModel(nn.Module):
    """GraphSAGE Model"""
    
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int,
                 num_layers: int = 3, dropout: float = 0.2):
        """
        Initialize GraphSAGE model
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            embedding_dim: Output embedding dimension
            num_layers: Number of SAGE layers
            dropout: Dropout rate
        """
        super(GraphSAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Final layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, embedding_dim))
        else:
            self.convs.append(SAGEConv(input_dim, embedding_dim))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Optional edge attributes (not used in GraphSAGE)
            
        Returns:
            Node embeddings [N, embedding_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x


class GNNModel(nn.Module):
    """Wrapper class for different GNN architectures"""
    
    def __init__(self, input_dim: int, config: dict):
        """
        Initialize GNN model based on config
        
        Args:
            input_dim: Input feature dimension
            config: Model configuration dictionary
        """
        super(GNNModel, self).__init__()
        
        architecture = config.get('architecture', 'gat').lower()
        hidden_dim = config.get('hidden_dim', 128)
        embedding_dim = config.get('embedding_dim', 64)
        num_layers = config.get('num_layers', 3)
        dropout = config.get('dropout', 0.2)
        
        if architecture == 'gat':
            num_heads = config.get('num_heads', 4)
            self.model = GATModel(input_dim, hidden_dim, embedding_dim,
                                 num_layers, num_heads, dropout)
        elif architecture == 'gcn':
            self.model = GCNModel(input_dim, hidden_dim, embedding_dim,
                                 num_layers, dropout)
        elif architecture == 'graphsage':
            self.model = GraphSAGEModel(input_dim, hidden_dim, embedding_dim,
                                      num_layers, dropout)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Optional edge attributes
            
        Returns:
            Node embeddings
        """
        return self.model(x, edge_index, edge_attr)

