"""
Graph Construction Module
Builds spatial graphs from preprocessed accessibility data
"""

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class GraphBuilder:
    """Builds spatial graphs for GNN training"""
    
    def __init__(self, config: Dict):
        """
        Initialize graph builder with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.graph_config = config['graph']
        
    def compute_geographic_distance(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """
        Compute geographic distance in meters using Haversine formula
        
        Args:
            coords1: Array of (lon, lat) coordinates
            coords2: Array of (lon, lat) coordinates
            
        Returns:
            Distance matrix in meters
        """
        # Earth radius in meters
        R = 6371000
        
        # Convert to radians
        lat1 = np.radians(coords1[:, 1])
        lat2 = np.radians(coords2[:, 1])
        lon1 = np.radians(coords1[:, 0])
        lon2 = np.radians(coords2[:, 0])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        distance = R * c
        return distance
    
    def build_knn_graph(self, coords: np.ndarray, features: np.ndarray, 
                        metadata: 'pd.DataFrame') -> Data:
        """
        Build KNN spatial graph
        
        Args:
            coords: Array of (lon, lat) coordinates
            features: Node feature matrix
            metadata: Metadata dataframe
            
        Returns:
            PyTorch Geometric Data object
        """
        k = self.graph_config['k_neighbors']
        print(f"Building KNN graph with k={k}")
        
        # Use geographic distance for KNN
        # For large datasets, use approximate nearest neighbors
        n_samples = len(coords)
        
        if n_samples > 10000:
            # Use approximate method for large datasets
            print("Using approximate KNN for large dataset")
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', 
                                   metric='haversine').fit(np.radians(coords))
            distances, indices = nbrs.kneighbors(np.radians(coords))
            # Remove self-connections
            indices = indices[:, 1:]
            distances = distances[:, 1:] * 6371000  # Convert to meters
        else:
            # Exact KNN for smaller datasets
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree',
                                   metric='haversine').fit(np.radians(coords))
            distances, indices = nbrs.kneighbors(np.radians(coords))
            indices = indices[:, 1:]
            distances = distances[:, 1:] * 6371000
        
        # Build edge list
        edge_list = []
        edge_attrs = []
        
        for i in range(n_samples):
            for j_idx, neighbor_idx in enumerate(indices[i]):
                edge_list.append([i, neighbor_idx])
                distance = distances[i, j_idx]
                
                # Compute edge features
                edge_attr = self._compute_edge_features(
                    i, neighbor_idx, features, metadata, distance
                )
                edge_attrs.append(edge_attr)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        
        # Node features
        x = torch.tensor(features, dtype=torch.float32)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        print(f"Graph built: {data.num_nodes} nodes, {data.num_edges} edges")
        
        return data
    
    def build_distance_threshold_graph(self, coords: np.ndarray, features: np.ndarray,
                                       metadata: 'pd.DataFrame') -> Data:
        """
        Build graph using distance threshold
        
        Args:
            coords: Array of (lon, lat) coordinates
            features: Node feature matrix
            metadata: Metadata dataframe
            
        Returns:
            PyTorch Geometric Data object
        """
        threshold = self.graph_config['distance_threshold']
        print(f"Building distance threshold graph with threshold={threshold}m")
        
        # Compute pairwise distances
        distances = self.compute_geographic_distance(coords, coords)
        
        # Find edges within threshold
        edge_list = []
        edge_attrs = []
        
        for i in range(len(coords)):
            neighbors = np.where((distances[i] <= threshold) & (distances[i] > 0))[0]
            for neighbor_idx in neighbors:
                edge_list.append([i, neighbor_idx])
                distance = distances[i, neighbor_idx]
                
                edge_attr = self._compute_edge_features(
                    i, neighbor_idx, features, metadata, distance
                )
                edge_attrs.append(edge_attr)
        
        if len(edge_list) == 0:
            raise ValueError("No edges found with given distance threshold")
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        
        x = torch.tensor(features, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        print(f"Graph built: {data.num_nodes} nodes, {data.num_edges} edges")
        
        return data
    
    def _compute_edge_features(self, i: int, j: int, features: np.ndarray,
                              metadata: 'pd.DataFrame', distance: float) -> np.ndarray:
        """
        Compute edge features between two nodes
        
        Args:
            i: Source node index
            j: Target node index
            features: Node feature matrix
            metadata: Metadata dataframe
            
        Returns:
            Edge feature vector
        """
        # Normalize distance (assuming max distance ~10km for normalization)
        distance_norm = min(distance / 10000.0, 1.0)
        
        # Label type similarity (check if same label type)
        label_type_i = metadata.iloc[i]['label_type']
        label_type_j = metadata.iloc[j]['label_type']
        type_similarity = 1.0 if label_type_i == label_type_j else 0.0
        
        # Severity difference
        severity_i = metadata.iloc[i]['severity']
        severity_j = metadata.iloc[j]['severity']
        severity_diff = abs(severity_i - severity_j) / 4.0  # Normalize to [0, 1]
        
        # Edge weight based on method
        weight_method = self.graph_config.get('edge_weight_method', 'inverse_distance')
        if weight_method == 'inverse_distance':
            weight = 1.0 / (1.0 + distance / 100.0)  # Inverse distance with smoothing
        elif weight_method == 'gaussian':
            sigma = self.graph_config.get('gaussian_sigma', 200.0)
            weight = np.exp(-(distance ** 2) / (2 * sigma ** 2))
        else:
            weight = 1.0
        
        # Combine edge features: [distance_norm, type_similarity, severity_diff, weight]
        edge_features = np.array([distance_norm, type_similarity, severity_diff, weight], 
                                 dtype=np.float32)
        
        return edge_features
    
    def build_graph(self, coords: np.ndarray, features: np.ndarray,
                   metadata: 'pd.DataFrame') -> Data:
        """
        Build graph using configured method
        
        Args:
            coords: Array of (lon, lat) coordinates
            features: Node feature matrix
            metadata: Metadata dataframe
            
        Returns:
            PyTorch Geometric Data object
        """
        method = self.graph_config['method']
        
        if method == 'knn':
            return self.build_knn_graph(coords, features, metadata)
        elif method == 'distance_threshold':
            return self.build_distance_threshold_graph(coords, features, metadata)
        else:
            raise ValueError(f"Unknown graph construction method: {method}")

