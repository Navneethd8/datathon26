"""
Hotspot Scoring Module
Computes multi-factor risk scores for spatial units
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


class HotspotScorer:
    """Compute risk scores for hotspot detection"""
    
    def __init__(self, config: Dict):
        """
        Initialize hotspot scorer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scoring_config = config['hotspot_scoring']
    
    def compute_density_score(self, metadata: pd.DataFrame, 
                              aggregation_metadata: pd.DataFrame) -> np.ndarray:
        """
        Compute density score (problems per unit area)
        
        Args:
            metadata: Original point metadata
            aggregation_metadata: Aggregated spatial unit metadata
            
        Returns:
            Density scores for each spatial unit
        """
        density_scores = []
        
        for _, unit in aggregation_metadata.iterrows():
            # Count points in this spatial unit
            if 'grid_id' in unit:
                # Grid-based: count points in grid cell
                count = unit['count']
                # Approximate area (assuming grid_size from config)
                grid_size = self.config['feature_extraction']['grid_size']
                area = grid_size ** 2  # square degrees (rough approximation)
            elif 'neighborhood' in unit:
                # Neighborhood-based: count points in neighborhood
                count = unit['count']
                # Approximate area (would need actual neighborhood boundaries)
                area = 1.0  # Normalize by count
            elif 'cluster_id' in unit:
                # Cluster-based: count points in cluster
                count = unit['count']
                area = 1.0  # Normalize by count
            else:
                count = 1
                area = 1.0
            
            density = count / max(area, 1e-6)
            density_scores.append(density)
        
        # Normalize to [0, 1]
        density_scores = np.array(density_scores)
        if density_scores.max() > density_scores.min():
            density_scores = (density_scores - density_scores.min()) / (density_scores.max() - density_scores.min())
        
        return density_scores
    
    def compute_severity_score(self, metadata: pd.DataFrame,
                              aggregation_metadata: pd.DataFrame) -> np.ndarray:
        """
        Compute severity score (average severity in area)
        
        Args:
            metadata: Original point metadata
            aggregation_metadata: Aggregated spatial unit metadata
            
        Returns:
            Severity scores for each spatial unit
        """
        severity_scores = []
        
        for _, unit in aggregation_metadata.iterrows():
            # Get points in this spatial unit
            if 'grid_id' in unit:
                # Match by grid
                grid_lon_idx = unit['grid_lon_idx']
                grid_lat_idx = unit['grid_lat_idx']
                grid_size = self.config['feature_extraction']['grid_size']
                min_lon = metadata['lon'].min()
                min_lat = metadata['lat'].min()
                
                unit_min_lon = min_lon + grid_lon_idx * grid_size
                unit_max_lon = unit_min_lon + grid_size
                unit_min_lat = min_lat + grid_lat_idx * grid_size
                unit_max_lat = unit_min_lat + grid_size
                
                mask = ((metadata['lon'] >= unit_min_lon) & (metadata['lon'] < unit_max_lon) &
                       (metadata['lat'] >= unit_min_lat) & (metadata['lat'] < unit_max_lat))
            elif 'neighborhood' in unit:
                mask = metadata['neighborhood'] == unit['neighborhood']
            elif 'cluster_id' in unit:
                # Would need cluster labels - approximate by distance
                center_lon, center_lat = unit['center_lon'], unit['center_lat']
                distances = np.sqrt((metadata['lon'] - center_lon)**2 + 
                                  (metadata['lat'] - center_lat)**2)
                # Use closest points
                threshold = np.percentile(distances, 10)  # Top 10% closest
                mask = distances <= threshold
            else:
                mask = np.zeros(len(metadata), dtype=bool)
            
            if mask.sum() > 0:
                unit_severities = metadata[mask]['severity'].values
                avg_severity = unit_severities.mean()
                # Normalize to [0, 1] (severity is 1-5)
                normalized_severity = (avg_severity - 1) / 4.0
            else:
                normalized_severity = 0.0
            
            severity_scores.append(normalized_severity)
        
        return np.array(severity_scores)
    
    def compute_diversity_score(self, metadata: pd.DataFrame,
                               aggregation_metadata: pd.DataFrame) -> np.ndarray:
        """
        Compute problem type diversity score (entropy-based)
        
        Args:
            metadata: Original point metadata
            aggregation_metadata: Aggregated spatial unit metadata
            
        Returns:
            Diversity scores for each spatial unit
        """
        diversity_scores = []
        label_types = ['SurfaceProblem', 'NoCurbRamp', 'CurbRamp']
        
        for _, unit in aggregation_metadata.iterrows():
            # Get points in this spatial unit (same logic as severity)
            if 'grid_id' in unit:
                grid_lon_idx = unit['grid_lon_idx']
                grid_lat_idx = unit['grid_lat_idx']
                grid_size = self.config['feature_extraction']['grid_size']
                min_lon = metadata['lon'].min()
                min_lat = metadata['lat'].min()
                
                unit_min_lon = min_lon + grid_lon_idx * grid_size
                unit_max_lon = unit_min_lon + grid_size
                unit_min_lat = min_lat + grid_lat_idx * grid_size
                unit_max_lat = unit_min_lat + grid_size
                
                mask = ((metadata['lon'] >= unit_min_lon) & (metadata['lon'] < unit_max_lon) &
                       (metadata['lat'] >= unit_min_lat) & (metadata['lat'] < unit_max_lat))
            elif 'neighborhood' in unit:
                mask = metadata['neighborhood'] == unit['neighborhood']
            elif 'cluster_id' in unit:
                center_lon, center_lat = unit['center_lon'], unit['center_lat']
                distances = np.sqrt((metadata['lon'] - center_lon)**2 + 
                                  (metadata['lat'] - center_lat)**2)
                threshold = np.percentile(distances, 10)
                mask = distances <= threshold
            else:
                mask = np.zeros(len(metadata), dtype=bool)
            
            if mask.sum() > 0:
                unit_labels = metadata[mask]['label_type'].values
                # Count each label type
                label_counts = [np.sum(unit_labels == label) for label in label_types]
                label_probs = np.array(label_counts) / len(unit_labels)
                # Compute entropy (higher entropy = more diversity)
                diversity = entropy(label_probs + 1e-10) / np.log(len(label_types))  # Normalize
            else:
                diversity = 0.0
            
            diversity_scores.append(diversity)
        
        return np.array(diversity_scores)
    
    def compute_combined_risk_score(self, density_scores: np.ndarray,
                                    severity_scores: np.ndarray,
                                    diversity_scores: np.ndarray,
                                    embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute combined risk score
        
        Args:
            density_scores: Density scores
            severity_scores: Severity scores
            diversity_scores: Diversity scores
            embeddings: Optional GNN embeddings
            
        Returns:
            Combined risk scores
        """
        alpha = self.scoring_config['density_weight']
        beta = self.scoring_config['severity_weight']
        gamma = self.scoring_config['diversity_weight']
        
        # Weighted combination
        risk_scores = (alpha * density_scores + 
                      beta * severity_scores + 
                      gamma * diversity_scores)
        
        # Optionally incorporate GNN embeddings
        if self.scoring_config.get('use_gnn_embeddings', False) and embeddings is not None:
            # Use embedding magnitude or distance from origin as additional signal
            embedding_magnitude = np.linalg.norm(embeddings, axis=1)
            # Normalize
            if embedding_magnitude.max() > embedding_magnitude.min():
                embedding_magnitude = ((embedding_magnitude - embedding_magnitude.min()) /
                                     (embedding_magnitude.max() - embedding_magnitude.min()))
            
            embedding_weight = self.scoring_config.get('embedding_weight', 0.3)
            risk_scores = (1 - embedding_weight) * risk_scores + embedding_weight * embedding_magnitude
        
        # Normalize to [0, 1]
        if risk_scores.max() > risk_scores.min():
            risk_scores = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min())
        
        return risk_scores
    
    def score(self, metadata: pd.DataFrame, aggregation_metadata: pd.DataFrame,
             embeddings: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Compute all risk scores
        
        Args:
            metadata: Original point metadata
            aggregation_metadata: Aggregated spatial unit metadata
            embeddings: Optional GNN embeddings
            
        Returns:
            Tuple of (risk_scores, score_components_dict)
        """
        density_scores = self.compute_density_score(metadata, aggregation_metadata)
        severity_scores = self.compute_severity_score(metadata, aggregation_metadata)
        diversity_scores = self.compute_diversity_score(metadata, aggregation_metadata)
        
        risk_scores = self.compute_combined_risk_score(
            density_scores, severity_scores, diversity_scores, embeddings
        )
        
        score_components = {
            'density': density_scores,
            'severity': severity_scores,
            'diversity': diversity_scores,
            'combined_risk': risk_scores
        }
        
        return risk_scores, score_components

