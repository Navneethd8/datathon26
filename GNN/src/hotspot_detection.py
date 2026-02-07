"""
Hotspot Detection Module
Uses clustering and spatial statistics to identify hotspots
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

try:
    from libpysal.weights import Queen, KNN
    from esda.getisord import G_Local
    HAS_SPATIAL_STATS = True
except ImportError:
    HAS_SPATIAL_STATS = False
    print("Warning: libpysal/esda not available. Spatial statistics disabled.")


class HotspotDetector:
    """Detect hotspots using clustering and spatial modeling"""
    
    def __init__(self, config: Dict):
        """
        Initialize hotspot detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.detection_config = config['hotspot_detection']
    
    def cluster_kmeans(self, features: np.ndarray, risk_scores: np.ndarray,
                      n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Cluster using K-means
        
        Args:
            features: Feature vectors (embeddings or risk scores)
            risk_scores: Risk scores
            n_clusters: Number of clusters (auto if None)
            
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            # Auto-determine: use elbow method or set to 5-10
            n_clusters = min(10, max(3, len(features) // 50))
        
        # Use risk scores for clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features.reshape(-1, 1))
        
        return labels
    
    def cluster_dbscan(self, features: np.ndarray, risk_scores: np.ndarray) -> np.ndarray:
        """
        Cluster using DBSCAN
        
        Args:
            features: Feature vectors
            risk_scores: Risk scores
            eps: DBSCAN eps parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            Cluster labels
        """
        eps = self.detection_config.get('dbscan_eps', 0.5)
        min_samples = self.detection_config.get('dbscan_min_samples', 10)
        
        # Use risk scores for DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(risk_scores.reshape(-1, 1))
        
        return labels
    
    def cluster_hierarchical(self, features: np.ndarray, risk_scores: np.ndarray,
                           n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Cluster using hierarchical clustering
        
        Args:
            features: Feature vectors
            risk_scores: Risk scores
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = min(10, max(3, len(features) // 50))
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(risk_scores.reshape(-1, 1))
        
        return labels
    
    def detect_hotspots_kde(self, coords: np.ndarray, risk_scores: np.ndarray,
                          bandwidth: float = 0.1) -> np.ndarray:
        """
        Detect hotspots using Kernel Density Estimation
        
        Args:
            coords: Coordinates [N, 2] (lon, lat)
            risk_scores: Risk scores
            bandwidth: KDE bandwidth
            
        Returns:
            Hotspot labels (1 = hotspot, 0 = not hotspot)
        """
        # Weight coordinates by risk scores
        weighted_coords = coords * risk_scores.reshape(-1, 1)
        
        # Fit KDE
        kde = gaussian_kde(weighted_coords.T, weights=risk_scores)
        
        # Evaluate KDE at each point
        kde_scores = kde(coords.T)
        
        # Threshold: top 20% are hotspots
        threshold = np.percentile(kde_scores, 80)
        hotspot_labels = (kde_scores >= threshold).astype(int)
        
        return hotspot_labels
    
    def detect_hotspots_getis_ord(self, coords: np.ndarray, risk_scores: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detect hotspots using Getis-Ord Gi* statistic
        
        Args:
            coords: Coordinates [N, 2]
            risk_scores: Risk scores
            
        Returns:
            Tuple of (hotspot_labels, gi_scores)
        """
        if not HAS_SPATIAL_STATS:
            print("Spatial statistics not available. Skipping Getis-Ord Gi*.")
            return np.zeros(len(coords), dtype=int), None
        
        try:
            # Create spatial weights matrix (KNN)
            k = min(10, len(coords) // 10)
            w = KNN.from_array(coords, k=k)
            
            # Compute Getis-Ord Gi* statistic
            gi = G_Local(risk_scores, w, permutations=99)
            
            # Get p-values
            alpha = self.detection_config.get('getis_ord_alpha', 0.05)
            
            # Hotspots: significant positive z-scores
            hotspot_labels = ((gi.Zs > 0) & (gi.p_norm < alpha)).astype(int)
            
            return hotspot_labels, gi.Zs
        except Exception as e:
            print(f"Error computing Getis-Ord Gi*: {e}")
            return np.zeros(len(coords), dtype=int), None
    
    def detect_hotspots(self, coords: np.ndarray, risk_scores: np.ndarray,
                       embeddings: Optional[np.ndarray] = None,
                       aggregation_metadata: Optional[pd.DataFrame] = None) -> Dict:
        """
        Detect hotspots using configured method
        
        Args:
            coords: Coordinates [N, 2] (lon, lat)
            risk_scores: Risk scores [N]
            embeddings: Optional embeddings [N, dim]
            aggregation_metadata: Optional metadata for spatial units
            
        Returns:
            Dictionary with hotspot labels and metadata
        """
        method = self.detection_config.get('clustering_method', 'dbscan')
        use_spatial_stats = self.detection_config.get('use_spatial_statistics', True)
        
        results = {}
        
        # Clustering-based detection
        if method == 'kmeans':
            n_clusters = self.detection_config.get('num_clusters')
            cluster_labels = self.cluster_kmeans(embeddings if embeddings is not None else risk_scores.reshape(-1, 1),
                                                risk_scores, n_clusters)
        elif method == 'dbscan':
            cluster_labels = self.cluster_dbscan(embeddings if embeddings is not None else risk_scores.reshape(-1, 1),
                                                risk_scores)
        elif method == 'hierarchical':
            n_clusters = self.detection_config.get('num_clusters')
            cluster_labels = self.cluster_hierarchical(embeddings if embeddings is not None else risk_scores.reshape(-1, 1),
                                                       risk_scores, n_clusters)
        else:
            # Default: threshold-based
            threshold = np.percentile(risk_scores, 80)  # Top 20%
            cluster_labels = (risk_scores >= threshold).astype(int)
        
        # Identify high-risk clusters as hotspots
        if len(np.unique(cluster_labels)) > 1:
            # Compute average risk per cluster
            cluster_risks = {}
            for cluster_id in np.unique(cluster_labels):
                if cluster_id >= 0:  # Ignore noise (-1)
                    mask = cluster_labels == cluster_id
                    cluster_risks[cluster_id] = risk_scores[mask].mean()
            
            # Top clusters are hotspots
            if cluster_risks:
                sorted_clusters = sorted(cluster_risks.items(), key=lambda x: x[1], reverse=True)
                n_hotspot_clusters = max(1, len(sorted_clusters) // 3)  # Top third
                hotspot_cluster_ids = {c[0] for c in sorted_clusters[:n_hotspot_clusters]}
                
                hotspot_labels = np.array([1 if c in hotspot_cluster_ids else 0 
                                         for c in cluster_labels])
            else:
                # Fallback: threshold-based
                threshold = np.percentile(risk_scores, 80)
                hotspot_labels = (risk_scores >= threshold).astype(int)
        else:
            # Single cluster or all noise: use threshold
            threshold = np.percentile(risk_scores, 80)
            hotspot_labels = (risk_scores >= threshold).astype(int)
        
        results['cluster_labels'] = cluster_labels
        results['hotspot_labels'] = hotspot_labels
        results['risk_scores'] = risk_scores
        
        # Spatial statistics
        if use_spatial_stats:
            # Getis-Ord Gi*
            gi_labels, gi_scores = self.detect_hotspots_getis_ord(coords, risk_scores)
            results['getis_ord_labels'] = gi_labels
            results['getis_ord_scores'] = gi_scores
            
            # KDE
            bandwidth = self.detection_config.get('kde_bandwidth', 0.1)
            kde_labels = self.detect_hotspots_kde(coords, risk_scores, bandwidth)
            results['kde_labels'] = kde_labels
            
            # Combine methods (majority vote)
            combined_labels = (hotspot_labels + gi_labels + kde_labels >= 2).astype(int)
            results['combined_hotspot_labels'] = combined_labels
        else:
            results['combined_hotspot_labels'] = hotspot_labels
        
        # Add metadata
        if aggregation_metadata is not None:
            aggregation_metadata = aggregation_metadata.copy()
            aggregation_metadata['risk_score'] = risk_scores
            aggregation_metadata['hotspot'] = results['combined_hotspot_labels']
            results['metadata'] = aggregation_metadata
        
        return results

