"""
Feature Extraction & Aggregation Module
Extracts GNN embeddings and aggregates them spatially
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

from .models.gnn_model import GNNModel


class FeatureExtractor:
    """Extract and aggregate GNN embeddings"""
    
    def __init__(self, model: GNNModel, config: Dict, device: Optional[torch.device] = None):
        """
        Initialize feature extractor
        
        Args:
            model: Trained GNN model
            config: Configuration dictionary
            device: Device to run on
        """
        self.model = model
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def extract_embeddings(self, data) -> np.ndarray:
        """
        Extract node embeddings from trained GNN
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Node embeddings [N, embedding_dim]
        """
        with torch.no_grad():
            data = data.to(self.device)
            embeddings = self.model(data.x, data.edge_index, data.edge_attr)
            return embeddings.cpu().numpy()
    
    def aggregate_grid(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                       grid_size: float) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Aggregate embeddings by grid cells
        
        Args:
            embeddings: Node embeddings [N, dim]
            metadata: Metadata dataframe with lon, lat
            grid_size: Grid cell size in degrees
            
        Returns:
            Tuple of (aggregated_embeddings, grid_metadata)
        """
        coords = metadata[['lon', 'lat']].values
        
        # Create grid cells
        min_lon, max_lon = coords[:, 0].min(), coords[:, 0].max()
        min_lat, max_lat = coords[:, 1].min(), coords[:, 1].max()
        
        # Compute grid indices
        grid_lon = ((coords[:, 0] - min_lon) / grid_size).astype(int)
        grid_lat = ((coords[:, 1] - min_lat) / grid_size).astype(int)
        
        # Create unique grid cell IDs
        grid_ids = grid_lon * 100000 + grid_lat  # Simple hash
        
        # Aggregate embeddings per grid cell (mean aggregation)
        unique_grids = np.unique(grid_ids)
        aggregated_embeddings = []
        grid_metadata_list = []
        
        for grid_id in unique_grids:
            mask = grid_ids == grid_id
            grid_embeddings = embeddings[mask]
            aggregated_emb = np.mean(grid_embeddings, axis=0)
            aggregated_embeddings.append(aggregated_emb)
            
            # Store grid cell center and count
            grid_coords = coords[mask]
            center_lon = grid_coords[:, 0].mean()
            center_lat = grid_coords[:, 1].mean()
            count = mask.sum()
            
            grid_metadata_list.append({
                'grid_id': grid_id,
                'center_lon': center_lon,
                'center_lat': center_lat,
                'count': count,
                'grid_lon_idx': grid_lon[mask][0],
                'grid_lat_idx': grid_lat[mask][0]
            })
        
        aggregated_embeddings = np.array(aggregated_embeddings)
        grid_metadata = pd.DataFrame(grid_metadata_list)
        
        print(f"Aggregated {len(embeddings)} nodes into {len(aggregated_embeddings)} grid cells")
        
        return aggregated_embeddings, grid_metadata
    
    def aggregate_neighborhood(self, embeddings: np.ndarray,
                              metadata: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Aggregate embeddings by neighborhood
        
        Args:
            embeddings: Node embeddings [N, dim]
            metadata: Metadata dataframe with neighborhood column
            
        Returns:
            Tuple of (aggregated_embeddings, neighborhood_metadata)
        """
        if 'neighborhood' not in metadata.columns:
            raise ValueError("Metadata must contain 'neighborhood' column")
        
        neighborhoods = metadata['neighborhood'].values
        
        # Aggregate per neighborhood
        unique_neighborhoods = np.unique(neighborhoods)
        aggregated_embeddings = []
        neighborhood_metadata_list = []
        
        for neighborhood in unique_neighborhoods:
            mask = neighborhoods == neighborhood
            neighborhood_embeddings = embeddings[mask]
            aggregated_emb = np.mean(neighborhood_embeddings, axis=0)
            aggregated_embeddings.append(aggregated_emb)
            
            # Store neighborhood info
            neighborhood_coords = metadata[mask][['lon', 'lat']].values
            center_lon = neighborhood_coords[:, 0].mean()
            center_lat = neighborhood_coords[:, 1].mean()
            count = mask.sum()
            
            neighborhood_metadata_list.append({
                'neighborhood': neighborhood,
                'center_lon': center_lon,
                'center_lat': center_lat,
                'count': count
            })
        
        aggregated_embeddings = np.array(aggregated_embeddings)
        neighborhood_metadata = pd.DataFrame(neighborhood_metadata_list)
        
        print(f"Aggregated {len(embeddings)} nodes into {len(aggregated_embeddings)} neighborhoods")
        
        return aggregated_embeddings, neighborhood_metadata
    
    def aggregate_density(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                         eps: float, min_samples: int) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Aggregate embeddings using DBSCAN density clustering
        
        Args:
            embeddings: Node embeddings [N, dim]
            metadata: Metadata dataframe
            eps: DBSCAN eps parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            Tuple of (aggregated_embeddings, cluster_metadata)
        """
        coords = metadata[['lon', 'lat']].values
        
        # Normalize coordinates for DBSCAN
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # Run DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(coords_scaled)
        
        # Aggregate per cluster
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]  # Remove noise (-1)
        
        aggregated_embeddings = []
        cluster_metadata_list = []
        
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_embeddings = embeddings[mask]
            aggregated_emb = np.mean(cluster_embeddings, axis=0)
            aggregated_embeddings.append(aggregated_emb)
            
            # Store cluster info
            cluster_coords = coords[mask]
            center_lon = cluster_coords[:, 0].mean()
            center_lat = cluster_coords[:, 1].mean()
            count = mask.sum()
            
            cluster_metadata_list.append({
                'cluster_id': cluster_id,
                'center_lon': center_lon,
                'center_lat': center_lat,
                'count': count
            })
        
        aggregated_embeddings = np.array(aggregated_embeddings)
        cluster_metadata = pd.DataFrame(cluster_metadata_list)
        
        print(f"Aggregated {len(embeddings)} nodes into {len(aggregated_embeddings)} clusters")
        
        return aggregated_embeddings, cluster_metadata
    
    def extract_and_aggregate(self, data, metadata: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extract embeddings and aggregate based on config
        
        Args:
            data: PyTorch Geometric Data object
            metadata: Metadata dataframe
            
        Returns:
            Tuple of (aggregated_embeddings, aggregation_metadata)
        """
        # Extract embeddings
        embeddings = self.extract_embeddings(data)
        
        # Aggregate based on method
        method = self.config['feature_extraction']['aggregation_method']
        
        if method == 'grid':
            grid_size = self.config['feature_extraction']['grid_size']
            return self.aggregate_grid(embeddings, metadata, grid_size)
        elif method == 'neighborhood':
            return self.aggregate_neighborhood(embeddings, metadata)
        elif method == 'density':
            eps = self.config['feature_extraction']['dbscan_eps']
            min_samples = self.config['feature_extraction']['dbscan_min_samples']
            return self.aggregate_density(embeddings, metadata, eps, min_samples)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

