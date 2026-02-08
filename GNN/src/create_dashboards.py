"""
Standalone script to create dashboards from saved results
Usage: python -m src.create_dashboards
"""

import json
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dashboard import ModelDashboard
from src.load_training_history import load_training_history


def main():
    """Create dashboards from saved results"""
    results_file = Path(__file__).parent.parent / 'outputs' / 'results.json'
    
    if not results_file.exists():
        print(f"ERROR: Results file not found at {results_file}")
        print("Please run the pipeline with --eval flag first to generate results.json")
        return
    
    print("Loading results...")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Found results from {results.get('timestamp', 'unknown time')}")
    
    # Create dashboard
    dashboard_dir = Path(__file__).parent.parent / 'outputs' / 'dashboards'
    dashboard = ModelDashboard(str(dashboard_dir))
    
    # Load training history from model checkpoint
    model_path = Path(__file__).parent.parent / 'models' / 'gnn_model.pt'
    history = load_training_history(str(model_path))
    if history is None:
        print("Warning: Could not load training history from model. Using empty history.")
        history = {'train_loss': []}
    
    # Get metrics
    metrics = results.get('evaluation', {})
    metrics['n_hotspots'] = results.get('n_hotspots')
    metrics['n_spatial_units'] = results.get('n_spatial_units')
    
    # Create dashboards
    dashboard.create_comprehensive_report(history, metrics, str(results_file))
    
    print("\n Dashboards created successfully!")


if __name__ == '__main__':
    main()

