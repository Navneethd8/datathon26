"""
Training Pipeline for GNN
Implements self-supervised, contrastive, and multi-task learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .models.gnn_model import GNNModel
from .graph_builder import GraphBuilder


class ContrastiveLoss(nn.Module):
    """Contrastive loss for self-supervised learning"""
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss
        
        Args:
            margin: Margin for contrastive loss
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings: torch.Tensor, positive_pairs: torch.Tensor,
                negative_pairs: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embeddings: Node embeddings [N, dim]
            positive_pairs: Positive pair indices [P, 2]
            negative_pairs: Negative pair indices [N, 2]
            
        Returns:
            Loss value
        """
        # Positive pair loss
        pos_emb1 = embeddings[positive_pairs[:, 0]]
        pos_emb2 = embeddings[positive_pairs[:, 1]]
        pos_dist = torch.norm(pos_emb1 - pos_emb2, dim=1)
        pos_loss = torch.mean(pos_dist ** 2)
        
        # Negative pair loss
        neg_emb1 = embeddings[negative_pairs[:, 0]]
        neg_emb2 = embeddings[negative_pairs[:, 1]]
        neg_dist = torch.norm(neg_emb1 - neg_emb2, dim=1)
        neg_loss = torch.mean(torch.clamp(self.margin - neg_dist, min=0) ** 2)
        
        return pos_loss + neg_loss


class ReconstructionLoss(nn.Module):
    """Reconstruction loss for self-supervised learning"""
    
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss
        
        Args:
            reconstructed: Reconstructed features
            original: Original features
            
        Returns:
            Loss value
        """
        return self.mse(reconstructed, original)


class GNNTrainer:
    """Trainer for GNN models"""
    
    def __init__(self, model: GNNModel, config: Dict, device: Optional[torch.device] = None):
        """
        Initialize trainer
        
        Args:
            model: GNN model
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )
        
        self.training_method = config['training']['method']
        
        # Initialize loss functions
        if self.training_method == 'contrastive':
            self.criterion = ContrastiveLoss(
                margin=config['training'].get('contrastive_margin', 1.0)
            )
        elif self.training_method == 'self_supervised':
            # Use reconstruction loss
            self.criterion = ReconstructionLoss()
            # Add decoder for reconstruction
            embedding_dim = self.model.embedding_dim
            input_dim = self.model.model.convs[0].in_channels
            self.decoder = nn.Linear(embedding_dim, input_dim).to(self.device)
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + list(self.decoder.parameters()),
                lr=config['training']['learning_rate']
            )
        else:  # multi_task
            self.criterion = nn.CrossEntropyLoss()
    
    def create_contrastive_pairs(self, data: Data, metadata: 'pd.DataFrame',
                                config: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create positive and negative pairs for contrastive learning
        
        Args:
            data: Graph data
            metadata: Metadata dataframe
            config: Training configuration
            
        Returns:
            Tuple of (positive_pairs, negative_pairs)
        """
        coords = metadata[['lon', 'lat']].values
        n_nodes = len(coords)
        
        # Use graph edges for positive pairs (they're already KNN-based, so nearby)
        edge_index = data.edge_index.cpu().numpy()
        edges = edge_index.T
        
        # Sample edges for positive pairs (limit to avoid too many)
        max_positive = min(10000, len(edges))
        sampled_edges = edges[:max_positive] if len(edges) <= max_positive else edges[np.random.choice(len(edges), max_positive, replace=False)]
        
        # Filter positive pairs: same label type (optional, can be relaxed)
        positive_pairs = []
        for edge in sampled_edges:
            i, j = edge[0], edge[1]
            # Optionally filter by same label type
            if metadata.iloc[i]['label_type'] == metadata.iloc[j]['label_type']:
                positive_pairs.append([i, j])
        
        # If not enough positive pairs, use all edges
        if len(positive_pairs) < 100:
            positive_pairs = sampled_edges[:min(5000, len(sampled_edges))].tolist()
        
        # For negative pairs, sample random pairs that are NOT connected
        # Create set of connected pairs for fast lookup
        connected_pairs = set()
        for edge in edges[:min(100000, len(edges))]:  # Limit for memory
            connected_pairs.add((edge[0], edge[1]))
            connected_pairs.add((edge[1], edge[0]))  # Undirected
        
        # Sample negative pairs
        n_negative = min(len(positive_pairs) * 2, 10000)  # Limit negative pairs
        negative_pairs = []
        max_attempts = n_negative * 10  # Try many times to find non-connected pairs
        attempts = 0
        
        while len(negative_pairs) < n_negative and attempts < max_attempts:
            i, j = np.random.choice(n_nodes, 2, replace=False)
            if (i, j) not in connected_pairs and (j, i) not in connected_pairs:
                negative_pairs.append([i, j])
            attempts += 1
        
        # If still not enough, just use random pairs (they're likely far apart)
        while len(negative_pairs) < n_negative:
            i, j = np.random.choice(n_nodes, 2, replace=False)
            negative_pairs.append([i, j])
        
        print(f"Created {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
        
        return (torch.tensor(positive_pairs, dtype=torch.long, device=device),
                torch.tensor(negative_pairs, dtype=torch.long, device=device))
    
    def train_epoch_contrastive(self, data: Data, metadata: 'pd.DataFrame') -> float:
        """Train one epoch with contrastive learning"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # For large graphs, sample a subset of nodes for training
        n_nodes = data.x.size(0)
        max_nodes = self.config['training'].get('max_nodes_per_batch', 20000)
        
        if n_nodes > max_nodes:
            # Sample nodes for training
            sample_size = max_nodes
            device = data.x.device
            node_indices = torch.randperm(n_nodes, device=device)[:sample_size]
            
            # Create subgraph - keep only edges where both endpoints are in sampled nodes
            edge_mask = torch.isin(data.edge_index[0], node_indices) & torch.isin(data.edge_index[1], node_indices)
            sub_edge_index = data.edge_index[:, edge_mask]
            
            # Map node indices to new indices (0 to sample_size-1)
            node_map = {int(old_idx.item()): int(new_idx) for new_idx, old_idx in enumerate(node_indices)}
            # Remap edge indices
            sub_edge_index_mapped = torch.zeros_like(sub_edge_index)
            for i in range(sub_edge_index.size(1)):
                sub_edge_index_mapped[0, i] = node_map[int(sub_edge_index[0, i].item())]
                sub_edge_index_mapped[1, i] = node_map[int(sub_edge_index[1, i].item())]
            
            sub_x = data.x[node_indices]
            sub_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else None
            
            # Get embeddings for subgraph
            embeddings = self.model(sub_x, sub_edge_index_mapped, sub_edge_attr)
            
            # Create pairs using sampled nodes
            sampled_metadata = metadata.iloc[node_indices.cpu().numpy()]
            pos_pairs, neg_pairs = self.create_contrastive_pairs_sampled(
                sub_edge_index_mapped, sampled_metadata, node_indices, node_map, self.config, device
            )
        else:
            # Process full graph
            embeddings = self.model(data.x, data.edge_index, data.edge_attr)
            pos_pairs, neg_pairs = self.create_contrastive_pairs(data, metadata, self.config, data.x.device)
        
        # Compute loss
        loss = self.criterion(embeddings, pos_pairs, neg_pairs)
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def compute_val_loss_contrastive(self, val_data: Data, val_metadata: 'pd.DataFrame') -> float:
        """Compute validation loss for contrastive learning"""
        self.model.eval()
        
        # Get embeddings
        embeddings = self.model(val_data.x, val_data.edge_index, val_data.edge_attr)
        
        # Create contrastive pairs
        pos_pairs, neg_pairs = self.create_contrastive_pairs(val_data, val_metadata, self.config, embeddings.device)
        
        # Compute loss
        loss = self.criterion(embeddings, pos_pairs, neg_pairs)
        
        return loss.item()
    
    def compute_val_loss_self_supervised(self, val_data: Data) -> float:
        """Compute validation loss for self-supervised learning"""
        self.model.eval()
        
        # Get embeddings
        embeddings = self.model(val_data.x, val_data.edge_index, val_data.edge_attr)
        
        # Reconstruct features
        reconstructed = self.decoder(embeddings)
        
        # Compute loss
        loss = self.criterion(reconstructed, val_data.x)
        
        return loss.item()
    
    def create_contrastive_pairs_sampled(self, edge_index: torch.Tensor, metadata: 'pd.DataFrame',
                                       node_indices: torch.Tensor, node_map: dict,
                                       config: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create contrastive pairs for sampled subgraph"""
        edges = edge_index.cpu().numpy().T
        n_nodes = len(node_indices)
        
        # Use edges for positive pairs
        max_positive = min(5000, len(edges))
        sampled_edges = edges[:max_positive] if len(edges) <= max_positive else edges[np.random.choice(len(edges), max_positive, replace=False)]
        
        positive_pairs = []
        for edge in sampled_edges:
            i, j = int(edge[0]), int(edge[1])
            if i < len(metadata) and j < len(metadata):
                if metadata.iloc[i]['label_type'] == metadata.iloc[j]['label_type']:
                    positive_pairs.append([i, j])
        
        if len(positive_pairs) < 100:
            positive_pairs = sampled_edges[:min(2000, len(sampled_edges))].tolist()
        
        # Negative pairs
        n_negative = min(len(positive_pairs) * 2, 5000)
        negative_pairs = []
        for _ in range(n_negative):
            i, j = np.random.choice(n_nodes, 2, replace=False)
            negative_pairs.append([int(i), int(j)])
        
        return (torch.tensor(positive_pairs, dtype=torch.long, device=device),
                torch.tensor(negative_pairs, dtype=torch.long, device=device))
    
    def train_epoch_self_supervised(self, data: Data) -> float:
        """Train one epoch with self-supervised reconstruction"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get embeddings
        embeddings = self.model(data.x, data.edge_index, data.edge_attr)
        
        # Reconstruct features
        reconstructed = self.decoder(embeddings)
        
        # Compute loss
        loss = self.criterion(reconstructed, data.x)
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, train_data: Data, train_metadata: 'pd.DataFrame',
             val_data: Optional[Data] = None, val_metadata: Optional['pd.DataFrame'] = None,
             num_epochs: Optional[int] = None,
             early_stopping_patience: Optional[int] = None) -> Dict:
        """
        Train the model
        
        Args:
            train_data: Training graph data
            train_metadata: Training metadata dataframe
            val_data: Optional validation graph data
            val_metadata: Optional validation metadata dataframe
            num_epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        num_epochs = num_epochs or self.config['training']['num_epochs']
        patience = early_stopping_patience or self.config['training'].get('early_stopping_patience', 10)
        
        # For very large graphs, use CPU to avoid OOM
        n_nodes = train_data.x.size(0)
        if n_nodes > 50000:
            print(f"Large graph detected ({n_nodes} nodes). Using CPU for training to avoid OOM.")
            print("  (Set max_nodes_per_batch in config to use GPU with subgraph sampling)")
            # Temporarily switch to CPU
            self.model.to('cpu')
            train_data = train_data.to('cpu')
            if val_data is not None:
                val_data = val_data.to('cpu')
            use_cpu = True
        else:
            # Move data to device
            train_data = train_data.to(self.device)
            if val_data is not None:
                val_data = val_data.to(self.device)
            use_cpu = False
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"Training method: {self.training_method}")
        print(f"Training samples: {train_data.num_nodes}, Validation samples: {val_data.num_nodes if val_data is not None else 0}")
        
        for epoch in tqdm(range(num_epochs), desc="Training"):
            # Training
            self.model.train()
            if self.training_method == 'contrastive':
                train_loss = self.train_epoch_contrastive(train_data, train_metadata)
            elif self.training_method == 'self_supervised':
                train_loss = self.train_epoch_self_supervised(train_data)
            else:
                # Multi-task would need labels - skip for now
                train_loss = 0.0
            
            history['train_loss'].append(train_loss)
            
            # Validation
            val_loss = None
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    if self.training_method == 'contrastive':
                        val_loss = self.compute_val_loss_contrastive(val_data, val_metadata)
                    elif self.training_method == 'self_supervised':
                        val_loss = self.compute_val_loss_self_supervised(val_data)
                
                if val_loss is not None:
                    history['val_loss'].append(val_loss)
            
            # Early stopping (disabled if patience is very high, e.g., 999999)
            if patience < 999999 and val_loss is not None:  # Only do early stopping if not disabled and validation available
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                # Track best loss even without early stopping
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
        
            if (epoch + 1) % 10 == 0:
                val_str = f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}{val_str}")
        
        # Move model back to original device if we switched to CPU
        if use_cpu and self.device.type == 'cuda':
            print("Moving model back to GPU for inference...")
            self.model.to(self.device)
        
        return history
    
    def save_model(self, path: str, history: Optional[Dict] = None):
        """Save model to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }
        if history:
            save_dict['history'] = history
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")

