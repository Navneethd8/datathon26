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
from .graph_builder import GraphBuilder
from .models.gnn_model import GNNModel
from .train import GNNTrainer
from .feature_extraction import FeatureExtractor
from .hotspot_scorer import HotspotScorer
from .hotspot_detection import HotspotDetector
from .evaluate import Evaluator
from .visualize import Visualizer


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
    print("\n" + "="*50)
    print("Step 1: Data Preprocessing")
    print("="*50)
    preprocessor = DataPreprocessor(config)
    features, metadata = preprocessor.process()
    print(f"Preprocessed {len(features)} data points")
    
    # Extract coordinates
    coords = metadata[['lon', 'lat']].values
    
    # 2. Graph Construction
    print("\n" + "="*50)
    print("Step 2: Graph Construction")
    print("="*50)
    graph_builder = GraphBuilder(config)
    graph_data = graph_builder.build_graph(coords, features, metadata)
    print(f"Graph constructed: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # 3. Model Setup
    print("\n" + "="*50)
    print("Step 3: Model Setup")
    print("="*50)
    input_dim = features.shape[1]
    model = GNNModel(input_dim, config['model'])
    print(f"Model created: {config['model']['architecture']}")
    
    # 4. Training
    trainer = GNNTrainer(model, config, device)
    
    model_path = args.model_path or Path(config['paths']['model_save_dir']) / 'gnn_model.pt'
    model_path = Path(__file__).parent.parent / model_path
    
    if args.train or not model_path.exists():
        print("\n" + "="*50)
        print("Step 4: Training")
        print("="*50)
        history = trainer.train(graph_data, metadata)
        trainer.save_model(str(model_path))
        print("Training complete!")
    else:
        print(f"\nLoading model from {model_path}")
        trainer.load_model(str(model_path))
    
    # 5. Feature Extraction
    print("\n" + "="*50)
    print("Step 5: Feature Extraction & Aggregation")
    print("="*50)
    feature_extractor = FeatureExtractor(model, config, device)
    aggregated_embeddings, aggregation_metadata = feature_extractor.extract_and_aggregate(
        graph_data, metadata
    )
    print(f"Extracted and aggregated embeddings: {aggregated_embeddings.shape}")
    
    # 6. Hotspot Scoring
    print("\n" + "="*50)
    print("Step 6: Hotspot Scoring")
    print("="*50)
    hotspot_scorer = HotspotScorer(config)
    risk_scores, score_components = hotspot_scorer.score(
        metadata, aggregation_metadata, aggregated_embeddings
    )
    print(f"Computed risk scores for {len(risk_scores)} spatial units")
    
    # 7. Hotspot Detection
    print("\n" + "="*50)
    print("Step 7: Hotspot Detection")
    print("="*50)
    hotspot_detector = HotspotDetector(config)
    aggregation_coords = aggregation_metadata[['center_lon', 'center_lat']].values
    hotspot_results = hotspot_detector.detect_hotspots(
        aggregation_coords, risk_scores, aggregated_embeddings, aggregation_metadata
    )
    n_hotspots = hotspot_results['combined_hotspot_labels'].sum()
    print(f"Detected {n_hotspots} hotspots")
    
    # 8. Evaluation
    metrics = {}
    if args.eval:
        print("\n" + "="*50)
        print("Step 8: Evaluation")
        print("="*50)
        evaluator = Evaluator(config)
        metrics = evaluator.evaluate(
            aggregated_embeddings,
            hotspot_results['cluster_labels'],
            aggregation_coords,
            risk_scores,
            hotspot_results['combined_hotspot_labels'],
            metadata
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
    
    # 9. Visualization
    if args.visualize:
        print("\n" + "="*50)
        print("Step 9: Visualization")
        print("="*50)
        output_dir = Path(config['paths']['output_dir'])
        output_dir = Path(__file__).parent.parent / output_dir
        visualizer = Visualizer(config, str(output_dir))
        visualizer.visualize(
            metadata, hotspot_results, aggregated_embeddings, aggregation_metadata
        )
    
    print("\n" + "="*50)
    print("Pipeline Complete!")
    print("="*50)


if __name__ == '__main__':
    main()

