"""
Data Splitting Utility
Handles train/validation/test splits for both node-level and grid-level data
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')


class DataSplitter:
    """Handles data splitting for GNN and baseline models"""
    
    def __init__(self, config: Dict):
        """
        Initialize data splitter
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.split_config = config.get('data_split', {})
        self.train_ratio = self.split_config.get('train_ratio', 0.7)
        self.val_ratio = self.split_config.get('val_ratio', 0.15)
        self.test_ratio = self.split_config.get('test_ratio', 0.15)
        self.random_state = self.split_config.get('random_state', 42)
        
        # Validate ratios sum to 1.0
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total}")
    
    def split_nodes(self, n_nodes: int) -> Dict[str, np.ndarray]:
        """
        Split node indices into train/val/test sets
        
        Args:
            n_nodes: Total number of nodes
            
        Returns:
            Dictionary with 'train', 'val', 'test' node indices
        """
        indices = np.arange(n_nodes)
        
        # First split: train vs (val + test)
        train_size = self.train_ratio
        val_test_size = self.val_ratio + self.test_ratio
        
        train_indices, val_test_indices = train_test_split(
            indices,
            test_size=val_test_size,
            random_state=self.random_state
        )
        
        # Second split: val vs test
        val_size = self.val_ratio / val_test_size if val_test_size > 0 else 0
        
        if val_size > 0 and len(val_test_indices) > 0:
            val_indices, test_indices = train_test_split(
                val_test_indices,
                test_size=(1 - val_size),
                random_state=self.random_state
            )
        else:
            val_indices = np.array([], dtype=int)
            test_indices = val_test_indices
        
        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
    
    def split_graph_data(self, graph_data: Data, metadata: pd.DataFrame) -> Tuple[Dict[str, Data], Dict[str, pd.DataFrame]]:
        """
        Split graph data into train/val/test sets
        
        Args:
            graph_data: Full graph data
            metadata: Node metadata
            
        Returns:
            Tuple of (split_graph_data, split_metadata)
        """
        n_nodes = graph_data.num_nodes
        splits = self.split_nodes(n_nodes)
        
        split_graphs = {}
        split_metadata = {}
        
        for split_name, indices in splits.items():
            # Get metadata for this split
            split_metadata[split_name] = metadata.iloc[indices].reset_index(drop=True)
            
            # Create subgraph for this split
            # Keep only edges where both endpoints are in the split
            edge_mask = (
                torch.isin(graph_data.edge_index[0], torch.tensor(indices, dtype=torch.long)) &
                torch.isin(graph_data.edge_index[1], torch.tensor(indices, dtype=torch.long))
            )
            
            sub_edge_index = graph_data.edge_index[:, edge_mask]
            
            # Map node indices to new indices (0 to len(indices)-1)
            node_map = {int(old_idx.item()): int(new_idx) for new_idx, old_idx in enumerate(torch.tensor(indices, dtype=torch.long))}
            
            # Remap edge indices
            sub_edge_index_mapped = torch.zeros_like(sub_edge_index)
            for i in range(sub_edge_index.size(1)):
                sub_edge_index_mapped[0, i] = node_map[int(sub_edge_index[0, i].item())]
                sub_edge_index_mapped[1, i] = node_map[int(sub_edge_index[1, i].item())]
            
            # Get node features and edge attributes for this split
            sub_x = graph_data.x[indices]
            sub_edge_attr = graph_data.edge_attr[edge_mask] if graph_data.edge_attr is not None else None
            
            # Create new Data object
            split_graphs[split_name] = Data(
                x=sub_x,
                edge_index=sub_edge_index_mapped,
                edge_attr=sub_edge_attr
            )
        
        return split_graphs, split_metadata
    
    def split_grid_data(self, grid_metadata: pd.DataFrame, 
                       grid_coords: np.ndarray,
                       grid_features: Optional[np.ndarray] = None) -> Dict:
        """
        Split grid-level data into train/val/test sets
        
        Args:
            grid_metadata: Grid cell metadata
            grid_coords: Grid cell coordinates
            grid_features: Optional grid features
            
        Returns:
            Dictionary with split data
        """
        n_grid_cells = len(grid_metadata)
        splits = self.split_nodes(n_grid_cells)
        
        result = {}
        
        for split_name, indices in splits.items():
            result[split_name] = {
                'metadata': grid_metadata.iloc[indices].reset_index(drop=True),
                'coords': grid_coords[indices],
                'indices': indices
            }
            
            if grid_features is not None:
                result[split_name]['features'] = grid_features[indices]
        
        return result
    
    def split_point_data(self, metadata: pd.DataFrame,
                        coords: np.ndarray,
                        features: Optional[np.ndarray] = None) -> Dict:
        """
        Split point-level data into train/val/test sets
        
        Args:
            metadata: Point metadata
            coords: Point coordinates
            features: Optional point features
            
        Returns:
            Dictionary with split data
        """
        n_points = len(metadata)
        splits = self.split_nodes(n_points)
        
        result = {}
        
        for split_name, indices in splits.items():
            result[split_name] = {
                'metadata': metadata.iloc[indices].reset_index(drop=True),
                'coords': coords[indices],
                'indices': indices
            }
            
            if features is not None:
                result[split_name]['features'] = features[indices]
        
        return result
    
    def print_split_summary(self, splits: Dict):
        """
        Print summary of data splits
        
        Args:
            splits: Dictionary with split data
        """
        print("\nData Split Summary")
        
        if 'train' in splits:
            if isinstance(splits['train'], dict):
                # Grid or point data
                for split_name in ['train', 'val', 'test']:
                    if split_name in splits:
                        n = len(splits[split_name]['metadata'])
                        pct = (n / sum(len(splits[s]['metadata']) for s in ['train', 'val', 'test'] if s in splits)) * 100
                        print(f"{split_name.capitalize()}: {n} samples ({pct:.1f}%)")
            else:
                # Node indices
                total = sum(len(splits[s]) for s in ['train', 'val', 'test'] if s in splits)
                for split_name in ['train', 'val', 'test']:
                    if split_name in splits:
                        n = len(splits[split_name])
                        pct = (n / total) * 100 if total > 0 else 0
                        print(f"{split_name.capitalize()}: {n} samples ({pct:.1f}%)")
        
        print("")

