"""
Baseline Models for Hotspot Detection
Implements classical spatial methods, clustering baselines, and ML baselines
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    from hdbscan import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("Warning: hdbscan not available. HDBSCAN baseline will be skipped.")

try:
    from libpysal.weights import Queen, KNN
    from esda.getisord import G_Local
    from esda.moran import Moran_Local
    HAS_SPATIAL_STATS = True
except ImportError:
    HAS_SPATIAL_STATS = False
    print("Warning: libpysal/esda not available. Spatial statistics baselines will be skipped.")

try:
    from node2vec import Node2Vec
    HAS_NODE2VEC = True
except ImportError:
    HAS_NODE2VEC = False
    print("Warning: node2vec not available. Node2Vec baseline will be skipped.")


class BaselineModels:
    """Implement baseline hotspot detection methods"""
    
    def __init__(self, config: Dict):
        """
        Initialize baseline models
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def kde_severity_weighted(self, coords: np.ndarray, severities: np.ndarray,
                             grid_coords: np.ndarray, bandwidth: float = 0.001,
                             top_percent: float = 4.22) -> Tuple[np.ndarray, np.ndarray]:
        """
        Severity-weighted KDE heatmap
        
        Args:
            coords: Point coordinates [N, 2]
            severities: Severity values [N]
            grid_coords: Grid cell centers [M, 2]
            bandwidth: KDE bandwidth
            top_percent: Top X% to mark as hotspots
            
        Returns:
            Tuple of (risk_surface, hotspot_labels)
        """
        # Weight coordinates by severity
        weights = severities / severities.max()  # Normalize
        
        # Fit KDE
        kde = gaussian_kde(coords.T, weights=weights, bw_method=bandwidth)
        
        # Evaluate on grid
        risk_surface = kde(grid_coords.T)
        
        # Normalize
        risk_surface = (risk_surface - risk_surface.min()) / (risk_surface.max() - risk_surface.min() + 1e-10)
        
        # Threshold top X%
        threshold = np.percentile(risk_surface, 100 - top_percent)
        hotspot_labels = (risk_surface >= threshold).astype(int)
        
        return risk_surface, hotspot_labels
    
    def getis_ord_gi_star(self, grid_coords: np.ndarray, cell_values: np.ndarray,
                         alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Getis-Ord Gi* local hotspot statistic
        
        Args:
            grid_coords: Grid cell centers [M, 2]
            cell_values: Values per cell (e.g., severity sum) [M]
            alpha: Significance level
            
        Returns:
            Tuple of (gi_scores, hotspot_labels)
        """
        if not HAS_SPATIAL_STATS:
            return np.zeros(len(grid_coords)), np.zeros(len(grid_coords), dtype=int)
        
        try:
            # Create spatial weights
            k = min(10, len(grid_coords) // 10)
            w = KNN.from_array(grid_coords, k=k)
            
            # Compute Gi*
            gi = G_Local(cell_values, w, permutations=99)
            
            # Hotspots: significant positive z-scores
            hotspot_labels = ((gi.Zs > 0) & (gi.p_norm < alpha)).astype(int)
            
            return gi.Zs, hotspot_labels
        except Exception as e:
            print(f"Error computing Getis-Ord Gi*: {e}")
            return np.zeros(len(grid_coords)), np.zeros(len(grid_coords), dtype=int)
    
    def local_morans_i(self, grid_coords: np.ndarray, cell_values: np.ndarray,
                      alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Local Moran's I / LISA clusters
        
        Args:
            grid_coords: Grid cell centers [M, 2]
            cell_values: Values per cell [M]
            alpha: Significance level
            
        Returns:
            Tuple of (lisa_scores, hotspot_labels)
        """
        if not HAS_SPATIAL_STATS:
            return np.zeros(len(grid_coords)), np.zeros(len(grid_coords), dtype=int)
        
        try:
            # Create spatial weights
            k = min(10, len(grid_coords) // 10)
            w = KNN.from_array(grid_coords, k=k)
            
            # Compute Local Moran's I
            lisa = Moran_Local(cell_values, w, permutations=99)
            
            # HH (high-high) clusters are hotspots
            hotspot_labels = ((lisa.q == 1) & (lisa.p_norm < alpha)).astype(int)
            
            return lisa.Is, hotspot_labels
        except Exception as e:
            print(f"Error computing Local Moran's I: {e}")
            return np.zeros(len(grid_coords)), np.zeros(len(grid_coords), dtype=int)
    
    def simple_thresholding(self, grid_features: np.ndarray, 
                           weights: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple thresholding on grid features (no GNN, no DBSCAN)
        
        Args:
            grid_features: Feature matrix [M, F] (density, severity, diversity, etc.)
            weights: Feature weights dict
            
        Returns:
            Tuple of (risk_scores, hotspot_labels)
        """
        if weights is None:
            weights = {'density': 0.4, 'severity': 0.4, 'diversity': 0.2}
        
        # Assume features are in order: [density, severity, diversity, ...]
        risk_scores = (weights.get('density', 0.4) * grid_features[:, 0] +
                      weights.get('severity', 0.4) * grid_features[:, 1] +
                      weights.get('diversity', 0.2) * grid_features[:, 2])
        
        # Normalize
        if risk_scores.max() > risk_scores.min():
            risk_scores = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min())
        
        # Threshold top 4.22% (matching GNN result)
        threshold = np.percentile(risk_scores, 100 - 4.22)
        hotspot_labels = (risk_scores >= threshold).astype(int)
        
        return risk_scores, hotspot_labels
    
    def hdbscan_clustering(self, features: np.ndarray, min_cluster_size: int = 10) -> np.ndarray:
        """
        HDBSCAN clustering
        
        Args:
            features: Feature matrix [M, F]
            min_cluster_size: Minimum cluster size
            
        Returns:
            Cluster labels
        """
        if not HAS_HDBSCAN:
            print("HDBSCAN not available, using DBSCAN instead")
            return self.dbscan_clustering(features)
        
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(features)
        
        return labels
    
    def optics_clustering(self, features: np.ndarray, min_samples: int = 10) -> np.ndarray:
        """
        OPTICS clustering
        
        Args:
            features: Feature matrix [M, F]
            min_samples: Minimum samples parameter
            
        Returns:
            Cluster labels
        """
        clusterer = OPTICS(min_samples=min_samples)
        labels = clusterer.fit_predict(features)
        
        return labels
    
    def dbscan_clustering(self, features: np.ndarray, eps: float = 0.5, 
                         min_samples: int = 10) -> np.ndarray:
        """
        DBSCAN clustering
        
        Args:
            features: Feature matrix [M, F]
            eps: Epsilon parameter
            min_samples: Minimum samples parameter
            
        Returns:
            Cluster labels
        """
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(features)
        
        return labels
    
    def kmeans_clustering(self, features: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """
        KMeans clustering
        
        Args:
            features: Feature matrix [M, F]
            n_clusters: Number of clusters (auto if None)
            
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = min(10, max(3, len(features) // 100))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        return labels
    
    def xgboost_anomaly_score(self, grid_features: np.ndarray, 
                              target_scores: np.ndarray,
                              train_indices: Optional[np.ndarray] = None,
                              test_indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        XGBoost/Random Forest anomaly score with train/test split
        
        Args:
            grid_features: Feature matrix [M, F]
            target_scores: Target risk scores (e.g., severity-weighted density) [M]
            train_indices: Optional training indices
            test_indices: Optional test indices
            
        Returns:
            Tuple of (predicted_scores, hotspot_labels)
        """
        # Use Random Forest (XGBoost requires separate install)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        if train_indices is not None and test_indices is not None:
            # Train/test split
            X_train = grid_features[train_indices]
            y_train = target_scores[train_indices]
            X_test = grid_features[test_indices]
            
            model.fit(X_train, y_train)
            predicted_scores_test = model.predict(X_test)
            
            # Create full predicted scores array
            predicted_scores = np.zeros(len(grid_features))
            predicted_scores[test_indices] = predicted_scores_test
            
            # Also predict on train for completeness (but use test for threshold)
            predicted_scores[train_indices] = model.predict(X_train)
        else:
            # No split - use all data (backward compatibility)
            model.fit(grid_features, target_scores)
            predicted_scores = model.predict(grid_features)
        
        # Normalize
        if predicted_scores.max() > predicted_scores.min():
            predicted_scores = (predicted_scores - predicted_scores.min()) / (predicted_scores.max() - predicted_scores.min())
        
        # Threshold top 4.22% (use test set if available)
        if test_indices is not None and len(test_indices) > 0:
            threshold = np.percentile(predicted_scores[test_indices], 100 - 4.22)
        else:
            threshold = np.percentile(predicted_scores, 100 - 4.22)
        
        hotspot_labels = (predicted_scores >= threshold).astype(int)
        
        return predicted_scores, hotspot_labels
    
    def autoencoder_anomaly_score(self, grid_features: np.ndarray,
                                 encoding_dim: int = 32,
                                 train_indices: Optional[np.ndarray] = None,
                                 test_indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Autoencoder on grid features with train/test split
        
        Args:
            grid_features: Feature matrix [M, F]
            encoding_dim: Encoding dimension
            train_indices: Optional training indices
            test_indices: Optional test indices
            
        Returns:
            Tuple of (reconstruction_error, hotspot_labels)
        """
        # Normalize features
        scaler = StandardScaler()
        
        if train_indices is not None and test_indices is not None:
            # Train/test split
            X_train = grid_features[train_indices]
            X_test = grid_features[test_indices]
            
            # Fit scaler on training data only
            scaler.fit(X_train)
            features_train_scaled = scaler.transform(X_train)
            features_test_scaled = scaler.transform(X_test)
        else:
            # No split - use all data (backward compatibility)
            features_scaled = scaler.fit_transform(grid_features)
            features_train_scaled = features_scaled
            features_test_scaled = features_scaled
            train_indices = np.arange(len(grid_features))
            test_indices = np.arange(len(grid_features))
        
        # Simple autoencoder using MLP
        input_dim = grid_features.shape[1]
        
        # Encoder
        encoder = MLPRegressor(hidden_layer_sizes=(64, encoding_dim), 
                              activation='relu', random_state=42, max_iter=200)
        encoder.fit(features_train_scaled, features_train_scaled)
        
        # Decoder (reverse)
        decoder = MLPRegressor(hidden_layer_sizes=(encoding_dim, 64, input_dim),
                              activation='relu', random_state=42, max_iter=200)
        
        # Get encoded representation
        encoded_train = encoder.predict(features_train_scaled)
        decoder.fit(encoded_train, features_train_scaled)
        
        # Reconstruct on test set
        encoded_test = encoder.predict(features_test_scaled)
        reconstructed_test = decoder.predict(encoded_test)
        
        # Reconstruction error as anomaly score
        reconstruction_error_test = np.mean((features_test_scaled - reconstructed_test) ** 2, axis=1)
        
        # Also compute on train for completeness
        reconstructed_train = decoder.predict(encoded_train)
        reconstruction_error_train = np.mean((features_train_scaled - reconstructed_train) ** 2, axis=1)
        
        # Create full reconstruction error array
        reconstruction_error = np.zeros(len(grid_features))
        reconstruction_error[test_indices] = reconstruction_error_test
        reconstruction_error[train_indices] = reconstruction_error_train
        
        # Normalize
        if reconstruction_error.max() > reconstruction_error.min():
            reconstruction_error = (reconstruction_error - reconstruction_error.min()) / \
                                 (reconstruction_error.max() - reconstruction_error.min())
        
        # Higher error = higher risk (inverse)
        risk_scores = 1 - reconstruction_error
        
        # Threshold top 4.22% (use test set if available)
        if test_indices is not None and len(test_indices) > 0:
            threshold = np.percentile(risk_scores[test_indices], 100 - 4.22)
        else:
            threshold = np.percentile(risk_scores, 100 - 4.22)
        
        hotspot_labels = (risk_scores >= threshold).astype(int)
        
        return risk_scores, hotspot_labels
    
    def node2vec_embeddings(self, edge_index, num_nodes: int, 
                           dimensions: int = 64) -> np.ndarray:
        """
        Node2Vec embeddings
        
        Args:
            edge_index: Edge indices [2, E]
            num_nodes: Number of nodes
            dimensions: Embedding dimension
            
        Returns:
            Node embeddings [N, dimensions]
        """
        if not HAS_NODE2VEC:
            print("Node2Vec not available. Using random embeddings.")
            return np.random.randn(num_nodes, dimensions)
        
        try:
            import networkx as nx
            # Convert to NetworkX graph
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            edges = edge_index.T.cpu().numpy()
            G.add_edges_from(edges)
            
            # Create Node2Vec model
            node2vec = Node2Vec(G, dimensions=dimensions, walk_length=30, 
                               num_walks=200, workers=4)
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            
            # Get embeddings
            embeddings = np.array([model.wv[str(i)] for i in range(num_nodes)])
            
            return embeddings
        except Exception as e:
            print(f"Error computing Node2Vec: {e}")
            return np.random.randn(num_nodes, dimensions)
    
    def compute_grid_cell_features(self, metadata: pd.DataFrame, 
                                   grid_metadata: pd.DataFrame,
                                   grid_size: float) -> np.ndarray:
        """
        Compute features for each grid cell
        
        Args:
            metadata: Original point metadata
            grid_metadata: Grid cell metadata
            grid_size: Grid cell size
            
        Returns:
            Feature matrix [M, F] where F = [density, severity, diversity]
        """
        features = []
        
        min_lon = metadata['lon'].min()
        min_lat = metadata['lat'].min()
        
        for _, grid_cell in grid_metadata.iterrows():
            # Find points in this grid cell
            grid_lon_idx = grid_cell.get('grid_lon_idx', 0)
            grid_lat_idx = grid_cell.get('grid_lat_idx', 0)
            
            cell_min_lon = min_lon + grid_lon_idx * grid_size
            cell_max_lon = cell_min_lon + grid_size
            cell_min_lat = min_lat + grid_lat_idx * grid_size
            cell_max_lat = cell_min_lat + grid_size
            
            mask = ((metadata['lon'] >= cell_min_lon) & (metadata['lon'] < cell_max_lon) &
                   (metadata['lat'] >= cell_min_lat) & (metadata['lat'] < cell_max_lat))
            
            if mask.sum() > 0:
                cell_data = metadata[mask]
                
                # Density (count per area)
                density = mask.sum() / (grid_size ** 2)
                
                # Average severity
                avg_severity = cell_data['severity'].mean() / 5.0  # Normalize
                
                # Diversity (entropy of label types)
                from scipy.stats import entropy
                label_types = ['SurfaceProblem', 'NoCurbRamp', 'CurbRamp']
                label_counts = [np.sum(cell_data['label_type'] == label) for label in label_types]
                label_probs = np.array(label_counts) / len(cell_data)
                diversity = entropy(label_probs + 1e-10) / np.log(len(label_types))
            else:
                density = 0.0
                avg_severity = 0.0
                diversity = 0.0
            
            features.append([density, avg_severity, diversity])
        
        return np.array(features)

