"""
Main Pipeline
Orchestrates the complete GNN hotspot detection pipeline
"""

import yaml
import argparse
from pathlib import Path
import torch
import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .data_preprocessing import DataPreprocessor
from .data_split import DataSplitter
from .graph_builder import GraphBuilder
from .models.gnn_model import GNNModel
from .train import GNNTrainer
from .feature_extraction import FeatureExtractor
from .hotspot_scorer import HotspotScorer
from .hotspot_detection import HotspotDetector
from .evaluate import Evaluator
from .visualize import Visualizer
from .dashboard import ModelDashboard
from .baseline_comparison import BaselineComparison


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='GNN Accessibility Hotspot Detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--train', action='store_true',
                       help='Train the model')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate the model')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print("Configuration loaded")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Preprocessing
    print("\nStep 1: Data Preprocessing")
    preprocessor = DataPreprocessor(config)
    features, metadata = preprocessor.process()
    print(f"Preprocessed {len(features)} data points")
    
    # Extract coordinates
    coords = metadata[['lon', 'lat']].values
    
    # 2. Graph Construction
    print("\nStep 2: Graph Construction")
    graph_builder = GraphBuilder(config)
    graph_data = graph_builder.build_graph(coords, features, metadata)
    print(f"Graph constructed: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # 2.5. Data Splitting
    print("\nStep 2.5: Data Splitting")
    data_splitter = DataSplitter(config)
    graph_splits, metadata_splits = data_splitter.split_graph_data(graph_data, metadata)
    
    # 3. Model Setup
    print("\nStep 3: Model Setup")
    input_dim = features.shape[1]
    model = GNNModel(input_dim, config['model'])
    print(f"Model created: {config['model']['architecture']}")
    
    # 4. Training
    trainer = GNNTrainer(model, config, device)
    
    model_path = args.model_path or Path(config['paths']['model_save_dir']) / 'gnn_model.pt'
    model_path = Path(__file__).parent.parent / model_path
    
    if args.train or not model_path.exists():
        # 4. Training
        print("\nStep 4: Training")
        history = trainer.train(
            graph_splits['train'], 
            metadata_splits['train'],
            val_data=graph_splits.get('val'),
            val_metadata=metadata_splits.get('val')
        )
        trainer.save_model(str(model_path), history)
        print("Training complete!")
    else:
        print(f"\nLoading model from {model_path}")
        trainer.load_model(str(model_path))
    
    # 5. Feature Extraction (on test set for evaluation)
    print("\nStep 5: Feature Extraction & Aggregation")
    feature_extractor = FeatureExtractor(model, config, device)
    
    # Extract embeddings on test set for evaluation
    test_aggregated_embeddings, test_aggregation_metadata = feature_extractor.extract_and_aggregate(
        graph_splits['test'], metadata_splits['test']
    )
    
    # Also extract on full graph for visualization/comparison
    full_aggregated_embeddings, full_aggregation_metadata = feature_extractor.extract_and_aggregate(
        graph_data, metadata
    )
    
    # Use test set for evaluation
    aggregated_embeddings = test_aggregated_embeddings
    aggregation_metadata = test_aggregation_metadata
    print(f"Extracted and aggregated embeddings: {aggregated_embeddings.shape} (test set)")
    
    # 6. Hotspot Scoring (on test set)
    print("\nStep 6: Hotspot Scoring (Test Set)")
    hotspot_scorer = HotspotScorer(config)
    risk_scores, score_components = hotspot_scorer.score(
        metadata_splits['test'], aggregation_metadata, aggregated_embeddings
    )
    print(f"Computed risk scores for {len(risk_scores)} spatial units")
    
    # 7. Hotspot Detection (on test set)
    print("\nStep 7: Hotspot Detection (Test Set)")
    hotspot_detector = HotspotDetector(config)
    aggregation_coords = aggregation_metadata[['center_lon', 'center_lat']].values
    hotspot_results = hotspot_detector.detect_hotspots(
        aggregation_coords, risk_scores, aggregated_embeddings, aggregation_metadata
    )
    n_hotspots = hotspot_results['combined_hotspot_labels'].sum()
    print(f"Detected {n_hotspots} hotspots")
    
    # 8. Evaluation (on test set)
    metrics = {}
    if args.eval:
        # 8. Evaluation (on test set)
        print("\nStep 8: Evaluation (Test Set)")
        evaluator = Evaluator(config)
        metrics = evaluator.evaluate(
            aggregated_embeddings,
            hotspot_results['cluster_labels'],
            aggregation_coords,
            risk_scores,
            hotspot_results['combined_hotspot_labels'],
            metadata_splits['test']  # Use test set metadata
        )
        evaluator.print_metrics(metrics)
        
        # Save results to JSON for summary update
        # Convert NumPy types to native Python types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert NumPy types to native Python types"""
            # Check if it's a NumPy type by checking the type name or using isinstance
            if type(obj).__module__ == 'numpy':
                if isinstance(obj, np.integer) or 'int' in str(type(obj)):
                    return int(obj)
                elif isinstance(obj, np.floating) or 'float' in str(type(obj)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif obj is None:
                return None
            return obj
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'num_epochs': len(history.get('train_loss', [])) if 'history' in locals() else None,
            'final_loss': float(history['train_loss'][-1]) if 'history' in locals() and history.get('train_loss') else None,
            'early_stopped': len(history.get('train_loss', [])) < config['training']['num_epochs'] if 'history' in locals() else None,
            'n_hotspots': int(hotspot_results['combined_hotspot_labels'].sum()),
            'n_spatial_units': int(len(risk_scores)),
            'evaluation': convert_to_native(metrics)
        }
        
        results_file = Path(__file__).parent.parent / 'outputs' / 'results.json'
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to {results_file}")
    
    # 9. Visualization (on full data for better visualization)
    if args.visualize:
        print("\nStep 9: Visualization (Full Data)")
        output_dir = Path(config['paths']['output_dir'])
        output_dir = Path(__file__).parent.parent / output_dir
        visualizer = Visualizer(config, str(output_dir))
        # Recompute on full data for visualization
        full_risk_scores, _ = hotspot_scorer.score(
            metadata, full_aggregation_metadata, full_aggregated_embeddings
        )
        full_hotspot_results = hotspot_detector.detect_hotspots(
            full_aggregation_metadata[['center_lon', 'center_lat']].values,
            full_risk_scores,
            full_aggregated_embeddings,
            full_aggregation_metadata
        )
        visualizer.visualize(
            metadata, full_hotspot_results, full_aggregated_embeddings, full_aggregation_metadata
        )
    
    # 10. Create Dashboards
    if args.eval:
        # 10. Create Dashboards
        print("\nStep 10: Creating Dashboards")
        dashboard_dir = Path(__file__).parent.parent / config['paths']['output_dir'] / 'dashboards'
        dashboard = ModelDashboard(str(dashboard_dir))
        
        # Add risk scores to metrics for dashboard
        if 'risk_scores' not in metrics:
            metrics['risk_scores'] = risk_scores.tolist() if hasattr(risk_scores, 'tolist') else list(risk_scores)
        
        dashboard.create_comprehensive_report(
            history if 'history' in locals() else {},
            metrics,
            str(Path(__file__).parent.parent / 'outputs' / 'results.json')
        )
    
    # 11. Baseline Comparison
    if args.eval:
        # 11. Baseline Comparison
        print("\nStep 11: Baseline Comparison")
        baseline_comparison = BaselineComparison(config)
        
        # Prepare GNN results (using test set results)
        gnn_results = {
            'hotspot_labels': hotspot_results['combined_hotspot_labels'],
            'risk_scores': risk_scores,
            'edge_index': graph_splits['test'].edge_index,  # Use test set graph
            'num_nodes': len(metadata_splits['test']),  # Use test set size
            'morans_i': metrics.get('morans_i')
        }
        
        # Split grid data for baseline comparisons
        grid_splits = data_splitter.split_grid_data(
            full_aggregation_metadata,  # Use full grid metadata for proper indexing
            full_aggregation_metadata[['center_lon', 'center_lat']].values,  # Full grid coords
            full_aggregated_embeddings  # Full embeddings
        )
        
        # Run baseline comparisons
        # Pass full grid metadata and full point metadata for proper feature computation
        comparison_results = baseline_comparison.compare_all_baselines(
            metadata_splits['test'],  # Use test set point metadata for evaluation
            full_aggregation_metadata,  # Use full grid metadata for proper indexing
            full_aggregation_metadata[['center_lon', 'center_lat']].values,  # Full grid coords
            gnn_results,
            train_indices=grid_splits['train']['indices'] if 'train' in grid_splits else None,
            test_indices=grid_splits['test']['indices'] if 'test' in grid_splits else None,
            full_metadata=metadata  # Pass full point metadata for grid feature computation
        )
        
        # Save comparison results
        comparison_file = Path(__file__).parent.parent / 'outputs' / 'baseline_comparison.json'
        comparison_file.parent.mkdir(exist_ok=True)
        
        # Convert to JSON-serializable format
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            return obj
        
        comparison_json = convert_for_json(comparison_results)
        with open(comparison_file, 'w') as f:
            json.dump(comparison_json, f, indent=2)
        print(f"\nComparison results saved to {comparison_file}")
        
        # Create comparison visualizations
        from .baseline_visualization import BaselineVisualizer
        baseline_viz = BaselineVisualizer()
        baseline_viz.visualize_all(comparison_results)
        
        # Update RESULTS_SUMMARY.md with baseline comparison
        print("\nUpdating RESULTS_SUMMARY.md with baseline comparison...")
        from .update_results_with_baselines import update_results_summary_with_baselines
        update_results_summary_with_baselines()
    
    print("\nPipeline Complete!")


if __name__ == '__main__':
    main()

