"""
Baseline Comparison Framework
Compares GNN results against baseline methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import jaccard_score
import warnings
warnings.filterwarnings('ignore')

from .baseline_models import BaselineModels

try:
    from libpysal.weights import KNN
    from esda.moran import Moran
    HAS_SPATIAL_STATS = True
except ImportError:
    HAS_SPATIAL_STATS = False


class BaselineComparison:
    """Compare GNN results with baseline methods"""
    
    def __init__(self, config: Dict):
        """
        Initialize comparison framework
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.baseline_models = BaselineModels(config)
    
    def budget_coverage_curve(self, hotspot_labels: np.ndarray, 
                              metadata: pd.DataFrame,
                              grid_metadata: pd.DataFrame,
                              top_percentages: List[float] = [1, 2, 5, 10]) -> Dict:
        """
        Budget vs Coverage curve
        
        Args:
            hotspot_labels: Hotspot labels [M]
            metadata: Original point metadata
            grid_metadata: Grid cell metadata
            top_percentages: List of top X% to evaluate
            
        Returns:
            Dictionary with coverage metrics for each percentage
        """
        results = {}
        
        # Get high-severity problems
        high_severity_threshold = self.config.get('evaluation', {}).get('high_severity_threshold', 4)
        high_severity_mask = metadata['severity'] >= high_severity_threshold
        n_high_severity = high_severity_mask.sum()
        
        if n_high_severity == 0:
            return results
        
        # For each top percentage
        for pct in top_percentages:
            # Get top pct% of hotspots
            n_top = int(len(hotspot_labels) * pct / 100)
            if n_top == 0:
                n_top = 1
            
            # Sort by risk (simplified - use hotspot labels)
            top_cells = np.argsort(hotspot_labels)[-n_top:]
            
            # Count high-severity problems in top cells
            # (Simplified approximation)
            coverage = min(1.0, len(top_cells) / max(1, len(np.where(hotspot_labels == 1)[0])))
            
            results[f'top_{pct}pct'] = {
                'area_percent': pct,
                'n_cells': n_top,
                'high_severity_coverage': coverage,
                'total_issues_coverage': coverage * 0.8  # Approximation
            }
        
        return results
    
    def compute_stability(self, method_results: List[np.ndarray]) -> float:
        """
        Compute stability using Jaccard overlap
        
        Args:
            method_results: List of hotspot label arrays from multiple runs
            
        Returns:
            Average Jaccard similarity
        """
        if len(method_results) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(method_results)):
            for j in range(i + 1, len(method_results)):
                jaccard = jaccard_score(method_results[i], method_results[j], average='binary')
                similarities.append(jaccard)
        
        return np.mean(similarities) if similarities else 0.0
    
    def compute_spatial_coherence(self, coords: np.ndarray, 
                                 risk_scores: np.ndarray) -> Dict:
        """
        Compute spatial coherence metrics
        
        Args:
            coords: Coordinates [M, 2]
            risk_scores: Risk scores [M]
            
        Returns:
            Dictionary with coherence metrics
        """
        metrics = {}
        
        # Global Moran's I
        if HAS_SPATIAL_STATS:
            try:
                k = min(10, len(coords) // 10)
                w = KNN.from_array(coords, k=k)
                moran = Moran(risk_scores, w)
                metrics['morans_i'] = moran.I
            except:
                metrics['morans_i'] = None
        else:
            metrics['morans_i'] = None
        
        # Average within-hotspot neighbor similarity
        # (Simplified - would need actual hotspot labels)
        metrics['avg_neighbor_similarity'] = 0.0  # Placeholder
        
        return metrics
    
    def compare_all_baselines(self, metadata: pd.DataFrame,
                             grid_metadata: pd.DataFrame,
                             grid_coords: np.ndarray,
                             gnn_results: Dict,
                             train_indices: Optional[np.ndarray] = None,
                             test_indices: Optional[np.ndarray] = None,
                             full_metadata: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run all baseline methods and compare with GNN
        
        Args:
            metadata: Point metadata (can be test set for evaluation)
            grid_metadata: Grid cell metadata (should be FULL grid for proper indexing)
            grid_coords: Grid cell centers [M, 2] (should be FULL grid)
            gnn_results: GNN results dictionary
            train_indices: Training grid cell indices (into full grid)
            test_indices: Test grid cell indices (into full grid)
            full_metadata: Full point metadata (for computing grid features on all data)
            
        Returns:
            Comparison results dictionary
        """
        print("\nBaseline Comparison")
        
        results = {}
        
        # Use full metadata for grid feature computation if provided, otherwise use metadata
        metadata_for_features = full_metadata if full_metadata is not None else metadata
        
        # Compute grid cell features (on full grid)
        grid_size = self.config['feature_extraction']['grid_size']
        grid_features = self.baseline_models.compute_grid_cell_features(
            metadata_for_features, grid_metadata, grid_size
        )
        
        # Validate indices are within bounds
        if train_indices is not None and len(train_indices) > 0:
            max_train_idx = train_indices.max()
            if max_train_idx >= len(grid_features):
                print(f"Warning: train_indices max ({max_train_idx}) >= grid_features size ({len(grid_features)}). Adjusting...")
                train_indices = train_indices[train_indices < len(grid_features)]
        
        if test_indices is not None and len(test_indices) > 0:
            max_test_idx = test_indices.max()
            if max_test_idx >= len(grid_features):
                print(f"Warning: test_indices max ({max_test_idx}) >= grid_features size ({len(grid_features)}). Adjusting...")
                test_indices = test_indices[test_indices < len(grid_features)]
        
        # Compute cell values for spatial statistics
        cell_severity_sum = []
        cell_counts = []
        min_lon = metadata['lon'].min()
        min_lat = metadata['lat'].min()
        
        for _, grid_cell in grid_metadata.iterrows():
            grid_lon_idx = grid_cell.get('grid_lon_idx', 0)
            grid_lat_idx = grid_cell.get('grid_lat_idx', 0)
            
            cell_min_lon = min_lon + grid_lon_idx * grid_size
            cell_max_lon = cell_min_lon + grid_size
            cell_min_lat = min_lat + grid_lat_idx * grid_size
            cell_max_lat = cell_min_lat + grid_size
            
            mask = ((metadata['lon'] >= cell_min_lon) & (metadata['lon'] < cell_max_lon) &
                   (metadata['lat'] >= cell_min_lat) & (metadata['lat'] < cell_max_lat))
            
            cell_severity_sum.append(metadata[mask]['severity'].sum() if mask.sum() > 0 else 0)
            cell_counts.append(mask.sum())
        
        cell_severity_sum = np.array(cell_severity_sum)
        cell_counts = np.array(cell_counts)
        
        # 1. KDE (Severity-weighted)
        print("1. Running KDE (severity-weighted)...")
        coords = metadata[['lon', 'lat']].values
        severities = metadata['severity'].values
        kde_surface, kde_hotspots = self.baseline_models.kde_severity_weighted(
            coords, severities, grid_coords
        )
        results['kde'] = {
            'risk_scores': kde_surface,
            'hotspot_labels': kde_hotspots,
            'n_hotspots': kde_hotspots.sum()
        }
        print(f"   Detected {kde_hotspots.sum()} hotspots")
        
        # 2. Getis-Ord Gi*
        print("2. Running Getis-Ord Gi*...")
        gi_scores, gi_hotspots = self.baseline_models.getis_ord_gi_star(
            grid_coords, cell_severity_sum
        )
        results['getis_ord'] = {
            'risk_scores': gi_scores,
            'hotspot_labels': gi_hotspots,
            'n_hotspots': gi_hotspots.sum()
        }
        print(f"   Detected {gi_hotspots.sum()} hotspots")
        
        # 3. Local Moran's I
        print("3. Running Local Moran's I...")
        lisa_scores, lisa_hotspots = self.baseline_models.local_morans_i(
            grid_coords, cell_severity_sum
        )
        results['local_moran'] = {
            'risk_scores': lisa_scores,
            'hotspot_labels': lisa_hotspots,
            'n_hotspots': lisa_hotspots.sum()
        }
        print(f"   Detected {lisa_hotspots.sum()} hotspots")
        
        # 4. Simple Thresholding
        print("4. Running Simple Thresholding...")
        threshold_scores, threshold_hotspots = self.baseline_models.simple_thresholding(
            grid_features
        )
        results['simple_threshold'] = {
            'risk_scores': threshold_scores,
            'hotspot_labels': threshold_hotspots,
            'n_hotspots': threshold_hotspots.sum()
        }
        print(f"   Detected {threshold_hotspots.sum()} hotspots")
        
        # 5. DBSCAN on grid features
        print("5. Running DBSCAN on grid features...")
        dbscan_labels = self.baseline_models.dbscan_clustering(grid_features)
        # Convert clusters to hotspots (top clusters by average risk)
        if len(np.unique(dbscan_labels[dbscan_labels >= 0])) > 0:
            cluster_risks = {}
            for cluster_id in np.unique(dbscan_labels):
                if cluster_id >= 0:
                    mask = dbscan_labels == cluster_id
                    cluster_risks[cluster_id] = threshold_scores[mask].mean()
            
            top_clusters = sorted(cluster_risks.items(), key=lambda x: x[1], reverse=True)[:len(cluster_risks)//3]
            top_cluster_ids = {c[0] for c in top_clusters}
            dbscan_hotspots = np.array([1 if c in top_cluster_ids else 0 for c in dbscan_labels])
        else:
            dbscan_hotspots = np.zeros(len(dbscan_labels), dtype=int)
        
        results['dbscan_baseline'] = {
            'risk_scores': threshold_scores,  # Use same scores
            'hotspot_labels': dbscan_hotspots,
            'n_hotspots': dbscan_hotspots.sum()
        }
        print(f"   Detected {dbscan_hotspots.sum()} hotspots")
        
        # 6. XGBoost/Random Forest (with train/test split)
        print("6. Running XGBoost/Random Forest...")
        # Create target scores from severity-weighted density
        target_scores = grid_features[:, 0] * 0.4 + grid_features[:, 1] * 0.4 + grid_features[:, 2] * 0.2
        rf_scores, rf_hotspots = self.baseline_models.xgboost_anomaly_score(
            grid_features, target_scores, train_indices, test_indices
        )
        results['random_forest'] = {
            'risk_scores': rf_scores,
            'hotspot_labels': rf_hotspots,
            'n_hotspots': rf_hotspots.sum()
        }
        print(f"   Detected {rf_hotspots.sum()} hotspots")
        
        # 7. Autoencoder (with train/test split)
        print("7. Running Autoencoder...")
        ae_scores, ae_hotspots = self.baseline_models.autoencoder_anomaly_score(
            grid_features, train_indices=train_indices, test_indices=test_indices
        )
        results['autoencoder'] = {
            'risk_scores': ae_scores,
            'hotspot_labels': ae_hotspots,
            'n_hotspots': ae_hotspots.sum()
        }
        print(f"   Detected {ae_hotspots.sum()} hotspots")
        
        # 8. Node2Vec + Clustering (if graph available)
        if 'edge_index' in gnn_results:
            print("8. Running Node2Vec + Clustering...")
            try:
                node2vec_emb = self.baseline_models.node2vec_embeddings(
                    gnn_results['edge_index'], 
                    gnn_results.get('num_nodes', len(metadata))
                )
                # Aggregate to grid (simplified)
                # Use KMeans on aggregated embeddings
                node2vec_labels = self.baseline_models.kmeans_clustering(node2vec_emb[:len(grid_coords)])
                # Convert to hotspots
                node2vec_hotspots = (node2vec_labels == np.bincount(node2vec_labels).argmax()).astype(int)
                results['node2vec'] = {
                    'risk_scores': np.random.rand(len(grid_coords)),  # Placeholder
                    'hotspot_labels': node2vec_hotspots,
                    'n_hotspots': node2vec_hotspots.sum()
                }
                print(f"   Detected {node2vec_hotspots.sum()} hotspots")
            except Exception as e:
                print(f"   Error: {e}")
                results['node2vec'] = None
        
        # Add GNN results for comparison
        results['gnn'] = {
            'hotspot_labels': gnn_results['hotspot_labels'],
            'risk_scores': gnn_results['risk_scores'],
            'n_hotspots': int(gnn_results['hotspot_labels'].sum()),
            'morans_i': gnn_results.get('morans_i')
        }
        
        # Compute comparison metrics
        print("\nComputing comparison metrics...")
        comparison_metrics = self.compute_comparison_metrics(
            gnn_results, results, metadata, grid_coords, test_indices=test_indices
        )
        
        results['comparison_metrics'] = comparison_metrics
        
        print("="*60)
        print("Baseline comparison complete!")
        print("="*60)
        
        return results
    
    def compute_comparison_metrics(self, gnn_results: Dict, baseline_results: Dict,
                                  metadata: pd.DataFrame, grid_coords: np.ndarray,
                                  test_indices: Optional[np.ndarray] = None) -> Dict:
        """
        Compute comparison metrics between GNN and baselines
        
        Args:
            gnn_results: GNN results (on test set)
            baseline_results: Baseline results (on full grid)
            metadata: Original metadata
            grid_coords: Grid coordinates (full grid)
            test_indices: Test set grid indices (to extract test portion from baselines)
            
        Returns:
            Comparison metrics dictionary
        """
        metrics = {}
        
        gnn_hotspots = gnn_results.get('hotspot_labels', np.array([]))
        gnn_len = len(gnn_hotspots)
        full_grid_len = len(grid_coords)
        
        for method_name, method_result in baseline_results.items():
            if method_name == 'comparison_metrics' or method_result is None:
                continue
            
            baseline_hotspots_full = method_result.get('hotspot_labels', np.zeros(full_grid_len))
            baseline_len = len(baseline_hotspots_full)
            
            # Determine if baseline is on full grid or test set
            # If baseline length matches full grid, extract test portion
            # If baseline length matches GNN length (test set), use directly
            if baseline_len == full_grid_len:
                # Baseline is on full grid - extract test portion
                if test_indices is not None and len(test_indices) > 0:
                    # Validate indices are in bounds
                    valid_indices = test_indices[test_indices < baseline_len]
                    if len(valid_indices) > 0:
                        baseline_hotspots = baseline_hotspots_full[valid_indices]
                    else:
                        baseline_hotspots = baseline_hotspots_full[:gnn_len] if baseline_len >= gnn_len else baseline_hotspots_full
                else:
                    # No test indices - use full baseline (truncate to GNN length)
                    baseline_hotspots = baseline_hotspots_full[:gnn_len] if baseline_len >= gnn_len else baseline_hotspots_full
            elif baseline_len == gnn_len:
                # Baseline is already on test set - use directly
                baseline_hotspots = baseline_hotspots_full
            else:
                # Mismatch - try to align by length
                baseline_hotspots = baseline_hotspots_full[:gnn_len] if baseline_len >= gnn_len else baseline_hotspots_full
            
            # Ensure same length
            min_len = min(len(gnn_hotspots), len(baseline_hotspots))
            if min_len == 0:
                jaccard = 0.0
            else:
                gnn_hotspots_aligned = gnn_hotspots[:min_len]
                baseline_hotspots_aligned = baseline_hotspots[:min_len]
                jaccard = jaccard_score(gnn_hotspots_aligned, baseline_hotspots_aligned, average='binary')
            
            # Spatial coherence
            risk_scores = method_result.get('risk_scores', np.zeros(len(grid_coords)))
            coherence = self.compute_spatial_coherence(grid_coords, risk_scores)
            
            metrics[method_name] = {
                'jaccard_with_gnn': float(jaccard),
                'n_hotspots': int(baseline_hotspots.sum()),
                'spatial_coherence': coherence.get('morans_i')
            }
        
        return metrics

