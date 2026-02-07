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
7. Visualizes results on interactive maps

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

**Full pipeline:**
```bash
python -m src.main --train --eval --visualize
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
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── graph_builder.py         # Spatial graph construction
│   ├── models/
│   │   └── gnn_model.py         # GNN architectures (GAT, GCN, GraphSAGE)
│   ├── train.py                 # Training pipeline
│   ├── feature_extraction.py    # Embedding extraction and aggregation
│   ├── hotspot_scorer.py       # Risk score computation
│   ├── hotspot_detection.py    # Clustering and spatial modeling
│   ├── evaluate.py             # Evaluation metrics
│   ├── visualize.py            # Interactive maps and plots
│   └── main.py                 # Main pipeline orchestration
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
└── GNN-README.md               # This file
```

## Output

The pipeline generates:

- **Model**: Trained GNN model saved to `models/gnn_model.pt`
- **Visualizations**: 
  - Interactive map: `outputs/hotspot_map.html`
  - Embeddings plot: `outputs/embeddings.png`
- **Evaluation Metrics**: Printed to console (silhouette score, Moran's I, coverage)

## Key Features

- **Multiple GNN Architectures**: GAT, GCN, and GraphSAGE support
- **Flexible Training**: Self-supervised, contrastive, or multi-task learning
- **Spatial Aggregation**: Grid-based, neighborhood-based, or density-based
- **Multi-factor Risk Scoring**: Combines density, severity, and problem diversity
- **Advanced Hotspot Detection**: K-means, DBSCAN, hierarchical clustering, Getis-Ord Gi*, KDE
- **Interactive Visualizations**: Folium/Plotly maps with hotspot overlays

## Dependencies

- PyTorch & PyTorch Geometric (GNN)
- NumPy, Pandas (data processing)
- Scikit-learn (clustering, preprocessing)
- GeoPandas, Shapely (spatial operations)
- Folium/Plotly (visualization)
- LibPySAL, ESDA (spatial statistics)

## Notes

- For large datasets (>10K points), the pipeline uses approximate KNN for efficiency
- Spatial statistics (Getis-Ord Gi*, Moran's I) require libpysal/esda
- Interactive maps work best with Folium or Plotly installed
- GPU acceleration is used automatically if available

## License

See main project README for license information.

