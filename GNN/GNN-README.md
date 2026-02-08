# GNN Accessibility Hotspot Detection

This project implements a Graph Neural Network (GNN) system for identifying high-risk accessibility hotspots in urban environments. The system uses a hybrid approach: GNN learns rich spatial-relationship features from accessibility point data, then clustering and spatial modeling identify hotspots based on combined density, severity, and problem type metrics.

## Overview

The pipeline processes accessibility data (curb ramps, surface problems, etc.) and:
1. Preprocesses the data and creates feature vectors
2. Constructs spatial graphs connecting nearby accessibility points
3. Trains a GNN to learn spatial-contextual patterns
4. Extracts embeddings and aggregates them spatially
5. Computes multi-factor risk scores
6. Detects hotspots using clustering and spatial statistics
7. Compares against baseline methods (KDE, Getis-Ord Gi*, Local Moran's I, etc.)
8. Generates comprehensive dashboards and visualizations
9. Automatically updates results summary with findings

## Installation

1. Install dependencies:
```bash
cd GNN
pip install -r requirements.txt
```

2. Ensure you have the data file at `../data/Access_to_Everyday_Life_Dataset.csv`

## Usage

### Basic Usage

Train the model and run the full pipeline:
```bash
python -m src.main --train --eval --visualize
```

### Command Line Options

- `--train`: Train the GNN model (required on first run)
- `--eval`: Run evaluation metrics
- `--visualize`: Create visualizations
- `--config`: Path to config file (default: config.yaml)
- `--model-path`: Path to saved model (default: models/gnn_model.pt)

### Example Workflows

**Train only:**
```bash
python -m src.main --train
```

**Evaluate existing model:**
```bash
python -m src.main --eval --model-path models/gnn_model.pt
```

**Full pipeline (with baselines and dashboards):**
```bash
python -m src.main --train --eval --visualize
```

**Update results summary after run:**
```bash
python update_summary.py
```

**Create dashboards from saved results:**
```bash
python -m src.create_dashboards
```

## Configuration

Edit `config.yaml` to customize:

- **Graph Construction**: KNN neighbors, distance thresholds
- **GNN Architecture**: Model type (GAT/GCN/GraphSAGE), layers, dimensions
- **Training**: Learning rate, epochs, training method (contrastive/self-supervised)
- **Feature Extraction**: Aggregation method (grid/neighborhood/density)
- **Hotspot Scoring**: Weights for density, severity, diversity
- **Hotspot Detection**: Clustering method, spatial statistics

## Project Structure

```
GNN/
├── src/
│   ├── data_preprocessing.py         # Data loading and preprocessing
│   ├── graph_builder.py              # Spatial graph construction
│   ├── models/
│   │   └── gnn_model.py              # GNN architectures (GAT, GCN, GraphSAGE)
│   ├── train.py                      # Training pipeline
│   ├── feature_extraction.py         # Embedding extraction and aggregation
│   ├── hotspot_scorer.py             # Risk score computation
│   ├── hotspot_detection.py          # Clustering and spatial modeling
│   ├── evaluate.py                   # Evaluation metrics
│   ├── visualize.py                  # Interactive maps and plots
│   ├── dashboard.py                 # Comprehensive dashboards
│   ├── baseline_models.py           # Baseline method implementations
│   ├── baseline_comparison.py        # Baseline comparison framework
│   ├── baseline_visualization.py     # Comparison visualizations
│   ├── update_results_summary.py    # Auto-update results summary
│   ├── update_results_with_baselines.py  # Update summary with baselines
│   ├── load_training_history.py      # Load training history from checkpoints
│   ├── create_dashboards.py         # Standalone dashboard creation
│   └── main.py                      # Main pipeline orchestration
├── config.yaml                       # Configuration file
├── requirements.txt                   # Python dependencies
├── update_summary.py                 # Quick summary update script
├── GNN-README.md                    # This file
├── RESULTS_SUMMARY.md                # Detailed results and findings
├── BASELINE_IMPLEMENTATION.md       # Baseline methods documentation
├── DASHBOARD_GUIDE.md               # Dashboard usage guide
└── HOW_TO_RERUN.md                  # Guide for rerunning experiments
```

## Output

The pipeline generates:

- **Model**: Trained GNN model saved to `models/gnn_model.pt` (includes training history)
- **Results**: 
  - `outputs/results.json` - All metrics and results in JSON format
  - `outputs/baseline_comparison.json` - Baseline comparison results
- **Visualizations**: 
  - Interactive map: `outputs/hotspot_map.html`
  - Embeddings plot: `outputs/embeddings.png`
  - Training curves: `outputs/dashboards/training_curves.png`
  - Evaluation dashboard: `outputs/dashboards/evaluation_dashboard.png`
  - Interactive dashboard: `outputs/dashboards/interactive_dashboard.html`
  - Baseline comparison: `outputs/baseline_comparison/comparison_table.png`
- **Documentation**:
  - `RESULTS_SUMMARY.md` - Automatically updated with results and baseline comparisons
- **Evaluation Metrics**: Printed to console (silhouette score, Moran's I, coverage)

## Key Features

### Core GNN Pipeline
- **Multiple GNN Architectures**: GAT, GCN, and GraphSAGE support
- **Flexible Training**: Self-supervised, contrastive, or multi-task learning
- **Spatial Aggregation**: Grid-based, neighborhood-based, or density-based
- **Multi-factor Risk Scoring**: Combines density, severity, and problem diversity
- **Advanced Hotspot Detection**: K-means, DBSCAN, hierarchical clustering, Getis-Ord Gi*, KDE

### Baseline Comparison System
- **Classical Spatial Methods**: KDE (severity-weighted), Getis-Ord Gi*, Local Moran's I, Simple Thresholding
- **Clustering Baselines**: HDBSCAN, OPTICS, DBSCAN, KMeans
- **ML Baselines**: XGBoost/Random Forest, Autoencoder, Node2Vec
- **Comprehensive Metrics**: Jaccard similarity, spatial coherence, budget vs coverage

### Visualization & Analysis
- **Interactive Visualizations**: Folium/Plotly maps with hotspot overlays
- **Comprehensive Dashboards**: Training curves, evaluation metrics, comparison tables
- **Automatic Documentation**: RESULTS_SUMMARY.md auto-updated with findings

### Advanced Features
- **Memory Efficient**: Automatic CPU fallback for large graphs, subgraph sampling
- **Early Stopping**: Configurable (disabled by default for full training)
- **Results Tracking**: JSON export for all metrics and comparisons

## Dependencies

### Core Dependencies
- PyTorch & PyTorch Geometric (GNN)
- NumPy, Pandas (data processing)
- Scikit-learn (clustering, preprocessing)
- GeoPandas, Shapely (spatial operations)
- Folium/Plotly (visualization)
- LibPySAL, ESDA (spatial statistics)

### Optional Dependencies (for baselines)
- `hdbscan` - HDBSCAN clustering baseline
- `node2vec` - Node2Vec graph embedding baseline
- `networkx` - Graph processing for Node2Vec

All dependencies listed in `requirements.txt`

## Documentation

- **RESULTS_SUMMARY.md**: Comprehensive results, metrics, and baseline comparisons
- **BASELINE_IMPLEMENTATION.md**: Details on all baseline methods
- **DASHBOARD_GUIDE.md**: Guide to using and customizing dashboards
- **HOW_TO_RERUN.md**: Instructions for rerunning experiments without early stopping

## Notes

- For large datasets (>10K points), the pipeline uses approximate KNN for efficiency
- Large graphs (>50K nodes) automatically use CPU to avoid GPU memory issues
- Spatial statistics (Getis-Ord Gi*, Moran's I) require libpysal/esda
- Some baseline methods require optional dependencies (hdbscan, node2vec)
- Interactive maps work best with Folium or Plotly installed
- GPU acceleration is used automatically if available
- Training history is saved in model checkpoint for dashboard generation

## Baseline Comparison

The system includes comprehensive baseline comparisons:

1. **Classical Spatial Methods**: KDE, Getis-Ord Gi*, Local Moran's I
2. **Clustering Baselines**: HDBSCAN, OPTICS, DBSCAN, KMeans
3. **ML Baselines**: Random Forest, Autoencoder, Node2Vec

All baselines are automatically compared against the GNN approach with metrics including:
- Jaccard similarity with GNN
- Spatial coherence (Moran's I)
- Hotspot detection counts
- Budget vs coverage curves

Results are automatically added to `RESULTS_SUMMARY.md` after each run.

## License

See main project README for license information.

