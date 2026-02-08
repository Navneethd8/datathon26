# Dashboard Guide

## Overview

The dashboard system provides comprehensive visualizations of model performance, training history, and evaluation metrics.

## Available Dashboards

### 1. Training Curves (`training_curves.png`)
- **Content**: Training loss over epochs
- **Features**:
  - Loss curve with moving average
  - Minimum loss annotation
  - Epoch markers

### 2. Evaluation Dashboard (`evaluation_dashboard.png`)
- **Content**: Comprehensive evaluation metrics
- **Sections**:
  - Silhouette Score
  - Moran's I (Spatial Autocorrelation)
  - High-Severity Coverage (pie chart)
  - Hotspot vs Non-Hotspot Comparison
  - Risk Score Distribution
  - Summary Statistics Table

### 3. Interactive Dashboard (`interactive_dashboard.html`)
- **Content**: Interactive Plotly visualizations
- **Features**:
  - Zoomable plots
  - Hover tooltips
  - Multiple subplots
  - Exportable images

## How to Generate Dashboards

### Option 1: Automatic (During Pipeline)
Run the pipeline with evaluation flag:
```bash
python -m src.main --train --eval --visualize
```
Dashboards are automatically created after evaluation.

### Option 2: Standalone (From Saved Results)
If you already have results saved:
```bash
python -m src.create_dashboards
```

This will:
1. Load results from `outputs/results.json`
2. Load training history from `models/gnn_model.pt`
3. Generate all dashboards in `outputs/dashboards/`

## Dashboard Locations

All dashboards are saved in:
```
GNN/outputs/dashboards/
├── training_curves.png
├── evaluation_dashboard.png
└── interactive_dashboard.html
```

## Viewing Dashboards

### Static Images (PNG)
- Open directly in any image viewer
- Embed in reports/presentations
- High resolution (150 DPI)

### Interactive Dashboard (HTML)
- Open `interactive_dashboard.html` in any web browser
- No server required
- Interactive features:
  - Zoom in/out
  - Pan
  - Hover for values
  - Download as PNG

## Dashboard Components

### Training Metrics
- **Loss Curve**: Shows training progress
- **Convergence**: Identifies when model stopped improving
- **Best Epoch**: Highlights minimum loss

### Evaluation Metrics
- **Silhouette Score**: Cluster quality (higher is better, range: -1 to 1)
- **Moran's I**: Spatial autocorrelation (positive = clustered, range: -1 to 1)
- **Coverage**: % of high-severity problems in hotspots
- **Risk Separation**: Difference between hotspot and non-hotspot risk

### Statistical Summaries
- Hotspot count and ratio
- Average risk scores
- Distribution statistics
- Comparison metrics

## Customization

To customize dashboards, edit `src/dashboard.py`:

1. **Change colors**: Modify color parameters in plotting functions
2. **Add metrics**: Extend `create_evaluation_metrics_dashboard()`
3. **Modify layout**: Adjust subplot gridspec
4. **Add plots**: Create new visualization methods

## Troubleshooting

### No Training History
If training history is missing:
- Check that model was saved with `--train` flag
- History is saved in model checkpoint
- Can manually create history dict if needed

### Missing Metrics
If metrics are missing:
- Ensure `--eval` flag was used
- Check `outputs/results.json` exists
- Verify evaluation completed successfully

### Plotly Not Available
If interactive dashboard fails:
- Install: `pip install plotly`
- Static dashboards will still work
- Interactive features will be skipped

## Best Practices

1. **Generate after each run**: Keep dashboards up-to-date
2. **Compare runs**: Save dashboards with timestamps for comparison
3. **Share results**: HTML dashboards are easy to share
4. **Document findings**: Use dashboards in reports/presentations

## Example Workflow

```bash
# 1. Run full pipeline
python -m src.main --train --eval --visualize

# 2. Dashboards auto-generated in outputs/dashboards/

# 3. View interactive dashboard
# Open outputs/dashboards/interactive_dashboard.html in browser

# 4. Use static images in reports
# Copy training_curves.png and evaluation_dashboard.png
```

## Advanced Usage

### Custom Dashboard Script
Create your own dashboard script:

```python
from src.dashboard import ModelDashboard
from src.load_training_history import load_training_history
import json

# Load data
with open('outputs/results.json') as f:
    results = json.load(f)
history = load_training_history('models/gnn_model.pt')

# Create dashboard
dashboard = ModelDashboard('custom_dashboards')
dashboard.create_comprehensive_report(history, results['evaluation'])
```

### Batch Processing
Generate dashboards for multiple runs:

```python
for run_dir in ['run1', 'run2', 'run3']:
    results_file = f'{run_dir}/outputs/results.json'
    model_file = f'{run_dir}/models/gnn_model.pt'
    # ... load and create dashboards
```

