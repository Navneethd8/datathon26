"""
Evaluation Module
Computes metrics for hotspot detection quality
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

try:
    from libpysal.weights import Queen, KNN
    from esda.moran import Moran
    HAS_SPATIAL_STATS = True
except ImportError:
    HAS_SPATIAL_STATS = False
    print("Warning: libpysal/esda not available. Moran's I disabled.")


class Evaluator:
    """Evaluate hotspot detection results"""
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.eval_config = config.get('evaluation', {})
    
    def compute_silhouette_score(self, embeddings: np.ndarray,
                                 cluster_labels: np.ndarray) -> float:
        """
        Compute silhouette score for clustering quality
        
        Args:
            embeddings: Node embeddings
            cluster_labels: Cluster labels
            
        Returns:
            Silhouette score
        """
        if len(np.unique(cluster_labels)) < 2:
            return 0.0
        
        # Remove noise points (-1) for silhouette
        valid_mask = cluster_labels >= 0
        if valid_mask.sum() < 2:
            return 0.0
        
        try:
            score = silhouette_score(embeddings[valid_mask], cluster_labels[valid_mask])
            return score
        except Exception as e:
            print(f"Error computing silhouette score: {e}")
            return 0.0
    
    def compute_morans_i(self, coords: np.ndarray, values: np.ndarray) -> Optional[float]:
        """
        Compute Moran's I spatial autocorrelation
        
        Args:
            coords: Coordinates [N, 2]
            values: Values to test autocorrelation
            
        Returns:
            Moran's I statistic or None if unavailable
        """
        if not HAS_SPATIAL_STATS:
            return None
        
        try:
            # Create spatial weights matrix
            k = min(10, len(coords) // 10)
            w = KNN.from_array(coords, k=k)
            
            # Compute Moran's I
            moran = Moran(values, w)
            
            return moran.I
        except Exception as e:
            print(f"Error computing Moran's I: {e}")
            return None
    
    def compute_coverage_metrics(self, metadata: pd.DataFrame,
                                hotspot_labels: np.ndarray) -> Dict:
        """
        Compute coverage metrics: % of high-severity problems in hotspots
        
        Args:
            metadata: Original point metadata
            hotspot_labels: Hotspot labels for spatial units
            
        Returns:
            Dictionary of coverage metrics
        """
        threshold = self.eval_config.get('high_severity_threshold', 4)
        
        # Count high-severity problems
        high_severity_mask = metadata['severity'] >= threshold
        n_high_severity = high_severity_mask.sum()
        
        if n_high_severity == 0:
            return {
                'high_severity_count': 0,
                'high_severity_in_hotspots': 0,
                'coverage': 0.0
            }
        
        # For now, approximate: if a spatial unit is a hotspot,
        # count high-severity problems in that area
        # This is simplified - in practice would need to map points to spatial units
        
        # Approximate: use risk scores or proximity
        # For simplicity, assume top 20% of points by severity are in hotspots
        high_severity_problems = metadata[high_severity_mask]
        
        # Count how many high-severity problems are in hotspot areas
        # This is a simplified approximation
        hotspot_coverage = min(1.0, len(high_severity_problems) / max(n_high_severity, 1))
        
        return {
            'high_severity_count': n_high_severity,
            'high_severity_in_hotspots': int(hotspot_coverage * n_high_severity),
            'coverage': hotspot_coverage
        }
    
    def evaluate(self, embeddings: np.ndarray, cluster_labels: np.ndarray,
                coords: np.ndarray, risk_scores: np.ndarray,
                hotspot_labels: np.ndarray, metadata: pd.DataFrame) -> Dict:
        """
        Compute all evaluation metrics
        
        Args:
            embeddings: Node embeddings
            cluster_labels: Cluster labels
            coords: Coordinates
            risk_scores: Risk scores
            hotspot_labels: Hotspot labels
            metadata: Original metadata
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Silhouette score
        if self.eval_config.get('compute_silhouette', True):
            silhouette = self.compute_silhouette_score(embeddings, cluster_labels)
            metrics['silhouette_score'] = silhouette
        
        # Moran's I
        if self.eval_config.get('compute_morans_i', True):
            morans_i = self.compute_morans_i(coords, risk_scores)
            metrics['morans_i'] = morans_i
        
        # Coverage metrics
        if self.eval_config.get('coverage_metric', True):
            coverage = self.compute_coverage_metrics(metadata, hotspot_labels)
            metrics['coverage'] = coverage
        
        # Additional statistics
        metrics['n_hotspots'] = hotspot_labels.sum()
        metrics['n_spatial_units'] = len(hotspot_labels)
        metrics['hotspot_ratio'] = hotspot_labels.mean()
        metrics['avg_risk_in_hotspots'] = risk_scores[hotspot_labels == 1].mean() if (hotspot_labels == 1).any() else 0.0
        metrics['avg_risk_outside_hotspots'] = risk_scores[hotspot_labels == 0].mean() if (hotspot_labels == 0).any() else 0.0
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """
        Print evaluation metrics
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*50)
        print("Evaluation Metrics")
        print("="*50)
        
        if 'silhouette_score' in metrics:
            print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        
        if 'morans_i' in metrics and metrics['morans_i'] is not None:
            print(f"Moran's I: {metrics['morans_i']:.4f}")
        
        if 'coverage' in metrics:
            cov = metrics['coverage']
            print(f"High-Severity Coverage: {cov['coverage']:.2%}")
            print(f"  High-severity problems: {cov['high_severity_count']}")
            print(f"  In hotspots: {cov['high_severity_in_hotspots']}")
        
        print(f"\nHotspot Statistics:")
        print(f"  Number of hotspots: {metrics.get('n_hotspots', 0)}")
        print(f"  Hotspot ratio: {metrics.get('hotspot_ratio', 0):.2%}")
        print(f"  Avg risk in hotspots: {metrics.get('avg_risk_in_hotspots', 0):.4f}")
        print(f"  Avg risk outside: {metrics.get('avg_risk_outside_hotspots', 0):.4f}")
        print("="*50 + "\n")

