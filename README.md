[![LinkedIn][swas-linkedin-shield]][swas-linkedin-url]
[![LinkedIn][navneeth-linkedin-shield]][navneeth-linkedin-url]

# Accessibility Analysis Datathon Project

This repository contains two complementary approaches to analyzing and predicting accessibility barriers in urban environments:

1. **GNN Hotspot Detection** - Graph Neural Network-based spatial hotspot identification
2. **Temporal Forecasting** - Transformer-based predictive modeling for accessibility trends

Both projects analyze the **Access to Everyday Life Dataset** (~82K accessibility barriers across Seattle neighborhoods) to support urban planning and infrastructure improvement decisions.

---

## Project Structure

```
datathon26/
├── data/                          # Shared dataset
│   └── Access_to_Everyday_Life_Dataset.csv
├── GNN/                          # Graph Neural Network Hotspot Detection
│   ├── src/                      # Source code
│   ├── outputs/                  # Results, visualizations, dashboards
│   ├── models/                   # Trained models
│   ├── dashboard.html           # Interactive dashboard
│   ├── RESULTS_SUMMARY.md       # Detailed results
│   └── GNN-README.md            # Project documentation
├── temporal_calc/                # Temporal Forecasting
│   ├── outputs/                  # Results, plots, forecasts
│   ├── checkpoints/              # Model checkpoints
│   ├── dashboard.html           # Interactive dashboard
│   └── transformer_README.md   # Project documentation
└── README.md                     # This file
```

---

## Project Overview

### 1. GNN Hotspot Detection (`/GNN`)

**Research Question:** *Identify high-risk accessibility hotspots using clustering and/or spatial modeling.*

**Approach:** Hybrid Graph Neural Network (GAT) that combines:
- **Spatial Modeling**: KNN-based graph construction, spatial autocorrelation analysis
- **Clustering**: DBSCAN on learned embeddings with multi-factor risk scoring
- **Graph-Based Learning**: Contrastive learning to capture spatial-contextual patterns

**Results (test set, 70/15/15 train/val/test split):**
- **255 hotspots identified** (3.74% of spatial units on test set)
- **100% coverage** of high-severity problems (3,634 issues with severity ≥ 4)
- **Spatial coherence**: Moran's I = 0.2376 (moderate positive spatial autocorrelation)
- **Baseline comparison**: Compared against 8 baseline methods (KDE, Getis-Ord Gi*, DBSCAN, etc.) with Jaccard similarity ranging from 0.01-0.05, indicating the GNN identifies different spatial patterns than classical methods

**Rationale for Graph Structure:** Accessibility barriers exhibit spatial dependencies. Nearby issues may share common causes (infrastructure age, neighborhood planning, terrain). Classical spatial statistics (KDE, Getis-Ord Gi*) treat locations as independent points, which may miss these contextual relationships. Graph structure explicitly models spatial neighborhoods, allowing the GNN to learn representations that encode these dependencies. However, this comes with computational cost and requires careful graph construction (KNN-based, k=15 in our implementation).

**[View GNN Dashboard](./GNN/dashboard.html)** | **[Read Full Results](./GNN/RESULTS_SUMMARY.md)**

---

### 2. Temporal Forecasting (`/temporal_calc`)

**Research Question:** *Predict future neighborhood accessibility scores based on historical barrier patterns.*

**Approach:** Transformer-based sequence-to-sequence model for multi-step forecasting:
- **Architecture**: Transformer encoder (6 layers, 12 heads, d_model=192) with sinusoidal positional encoding
- **Features**: Temporal lags (t-1, t-2, t-3), rolling statistics (3 and 5-period), cyclical time encoding (sin/cos), demographic equity scores
- **Training**: Multi-step prediction (3 steps ahead) with Huber loss. Data split: temporal train/test split (not random, to preserve temporal structure)
- **Limitation**: Dataset lacks true timestamps; synthetic time bins created from `attribute_id` ordering as temporal proxy, which may not reflect actual temporal dynamics

**Results (test set, temporal train/test split):**
- **R² = 0.8372** (Transformer, test set)
- **MAE = 16.01 points** (vs. LSTM: 19.13, XGBoost: 40.69 on same test set)
- **55.56% threshold accuracy** (±15 points on accessibility score scale)
- **Model comparison**: Transformer achieves higher R² than LSTM (0.7893) and XGBoost (0.1008) on the test set. Note: XGBoost's poor performance (R² = 0.10) suggests it struggles with the temporal structure, while LSTM and Transformer both capture temporal dependencies, with Transformer showing better fit.

**Model Comparison:**
| Model | MAE | R² | Pct Accuracy | Threshold Accuracy |
|-------|-----|----|--------------|-------------------|
| **Transformer** | **16.01** | **0.8372** | **57.92%** | **55.56%** |
| LSTM | 19.13 | 0.7893 | 60.67% | 48.15% |
| XGBoost | 40.69 | 0.1008 | -23.91% | 0.00% |

**[View Temporal Dashboard](./temporal_calc/dashboard.html)** | **[Read Full Documentation](./temporal_calc/transformer_README.md)**

---

## How the Projects Complement Each Other

| Aspect | GNN Hotspot Detection | Temporal Forecasting |
|--------|----------------------|---------------------|
| **Goal** | Identify **where** high-risk areas are now | Predict **how** accessibility will change |
| **Time Horizon** | Current state analysis | Future trend prediction |
| **Output** | Spatial hotspot map (255 regions) | Neighborhood risk forecasts |
| **Use Case** | Prioritize infrastructure investment | Plan long-term interventions |
| **Method** | Graph-based spatial clustering | Sequence-to-sequence forecasting |

**Together, they provide:**
1. **Spatial Analysis**: Current spatial distribution of high-risk areas (GNN)
2. **Temporal Projections**: Forecasted accessibility trends (Temporal), with caveats about synthetic time bins
3. **Complementary Information**: Combining current hotspot locations with predicted trends may inform resource allocation, though both models have limitations (GNN: parameter sensitivity, Temporal: synthetic time proxy) that should be considered in decision-making

---

## Quick Start

### GNN Hotspot Detection

```bash
cd GNN

# Install dependencies
pip install -r requirements.txt

# Run full pipeline (train, evaluate, visualize)
python -m src.main --train --eval --visualize

# Generate map data for dashboard
python generate_map_data.py

# View dashboard (start local server)
python -m http.server 8000
# Open http://localhost:8000/dashboard.html
```

**📖 [Full GNN Documentation](./GNN/GNN-README.md)**

### Temporal Forecasting

```bash
cd temporal_calc

# Install dependencies
pip install torch numpy pandas scikit-learn xgboost matplotlib seaborn

# Run full pipeline
bash run_all.sh

# Or run individual steps:
python max_performance.py    # Train and compare models
python evaluate.py           # Evaluate best model
python generate_map_data.py  # Generate dashboard data

# View dashboard (start local server)
python -m http.server 8000
# Open http://localhost:8000/dashboard.html
```

**📖 [Full Temporal Documentation](./temporal_calc/transformer_README.md)**

---

## Key Findings Summary

### GNN Hotspot Detection Findings

1. **Detection Coverage**: Identifies 255 hotspots on test set (3.3x more than best baseline method) while maintaining 100% coverage of high-severity issues. This suggests the GNN captures a broader set of high-risk areas compared to conservative methods like Getis-Ord Gi* (17 hotspots).
2. **Pattern Divergence**: Low Jaccard similarity (0.01-0.05) with all baselines indicates the GNN identifies different spatial patterns than classical methods. This may reflect learned spatial-contextual relationships, though it could also indicate overfitting or parameter sensitivity. Further validation would be needed to confirm generalizability.
3. **Spatial Coherence Tradeoff**: Moderate spatial coherence (Moran's I = 0.2376) suggests meaningful spatial clustering without being overly conservative. Getis-Ord Gi* achieves higher coherence (0.75) but identifies far fewer hotspots (17), illustrating the tradeoff between coverage and spatial coherence.
4. **Graph Structure Justification**: Graph structure enables modeling spatial dependencies that point-based methods (KDE, thresholding) cannot capture. However, this requires careful graph construction (KNN k=15 chosen empirically) and comes with computational overhead. The necessity claim is supported by low baseline Jaccard similarity, but alternative graph constructions (e.g., radius-based) were not exhaustively explored due to 24-hour datathon constraints.

### Temporal Forecasting Findings

1. **Model Performance**: Transformer achieves R² = 0.8372 on test set, higher than LSTM (0.7893) and XGBoost (0.1008). The improvement over LSTM is modest (~6% relative), while XGBoost's poor performance (R² = 0.10) suggests it fails to capture temporal dependencies. The Transformer's attention mechanism may better model long-range dependencies in the sequence, though this comes with increased model complexity (~71K parameters vs. simpler baselines).
2. **Multi-step Forecasting**: Forecasts 3 steps ahead with 55.56% threshold accuracy (±15 points on accessibility score scale). This indicates the model captures general trends but has limited precision for point predictions. The model is appropriate for identifying neighborhoods at risk rather than exact score prediction.
3. **Feature Engineering Impact**: Temporal lags, rolling statistics, and cyclical encoding provide the model with historical context and periodic patterns. Ablation studies were not conducted within the 24-hour constraint to quantify each feature's contribution.
4. **Limitations**: Predictions are based on synthetic time bins (derived from `attribute_id` ordering), not true temporal data. This limits the model's ability to capture real-world temporal dynamics. Additionally, the dataset's crowdsourced nature may introduce reporting biases that affect generalizability.

---

## Results Comparison

### Spatial Analysis (GNN)
- **Hotspots Detected**: 255
- **Spatial Coverage**: 3.74% of units
- **High-Severity Coverage**: 100% (3,634 issues)
- **Spatial Coherence**: Moran's I = 0.2376

### Temporal Analysis (Temporal)
- **Prediction Accuracy**: R² = 0.8372
- **Mean Absolute Error**: 16.01 points
- **Threshold Accuracy**: 55.56% (±15 points)
- **Forecast Horizon**: 3 steps ahead

---

## Technical Stack

### GNN Project
- **Deep Learning**: PyTorch, PyTorch Geometric
- **Spatial Analysis**: scikit-learn, scipy, libpysal
- **Visualization**: Folium, Plotly, Matplotlib
- **Clustering**: DBSCAN, HDBSCAN, OPTICS

### Temporal Project
- **Deep Learning**: PyTorch (Transformer architecture)
- **Time Series**: Custom sequence-to-sequence model
- **Baselines**: LSTM, XGBoost
- **Visualization**: Matplotlib, Seaborn, Leaflet

---

## Output Files

### GNN Outputs (`/GNN/outputs/`)
- `hotspot_map.html` - Interactive map of detected hotspots
- `results.json` - Evaluation metrics and results
- `baseline_comparison.json` - Comparison with 8 baseline methods
- `embeddings.png` - Learned embedding visualization
- `dashboards/` - Training curves, evaluation dashboards
- `map_data.json` - Data for interactive dashboard

### Temporal Outputs (`/temporal_calc/outputs/`)
- `model_results.json` - Model performance metrics
- `neighborhood_risk_forecast.csv` - Future accessibility predictions
- `accessibility_scores.csv` - Current neighborhood scores
- `plots/` - Performance comparisons, geospatial analysis
- `map_data.json` - Data for interactive dashboard
- `eval/` - Detailed evaluation metrics and plots

---

## Research Contributions

### GNN Project
- Evaluates whether graph structure improves spatial dependency modeling compared to point-based methods (baseline comparison suggests it identifies different patterns, though necessity is not definitively proven)
- Implements a hybrid approach (spatial modeling + clustering + graph learning) and compares against 8 baseline methods using Jaccard similarity and spatial coherence metrics
- Provides baseline comparison (8 methods) with interpretable metrics (Jaccard similarity, Moran's I, coverage)
- Achieves 100% coverage of high-severity issues (severity ≥ 4) on test set, though hotspot boundaries depend on DBSCAN clustering parameters (eps, min_samples) which were selected empirically

### Temporal Project
- Compares Transformer, LSTM, and XGBoost architectures for accessibility forecasting (Transformer shows higher R² on test set, though improvement over LSTM is modest)
- Implements multi-step prediction (3 steps ahead) with 55.56% threshold accuracy (±15 points), indicating trend-level rather than point-prediction accuracy
- Integrates temporal features (lags, rolling stats, cyclical encoding) and demographic equity scores, though feature importance analysis was not conducted within the 24-hour constraint
- Provides forecast outputs that may inform planning, with important caveats: predictions are based on synthetic time bins (not true temporal data) and crowdsourced data may have reporting biases

---

## Documentation

- **[GNN Project README](./GNN/GNN-README.md)** - Complete GNN documentation
- **[GNN Results Summary](./GNN/RESULTS_SUMMARY.md)** - Detailed results and analysis
- **[GNN Dashboard Guide](./GNN/DASHBOARD_GUIDE.md)** - Dashboard usage guide
- **[Temporal Project README](./temporal_calc/transformer_README.md)** - Complete temporal documentation
- **[Temporal Data Loading Guide](./temporal_calc/DATA_LOADING_GUIDE.md)** - Data preparation guide

---

## Use Cases

### For Urban Planners
- **GNN**: Identify priority areas for immediate infrastructure investment
- **Temporal**: Forecast which neighborhoods will need future interventions

### For Policy Makers
- **GNN**: Understand current spatial distribution of accessibility barriers
- **Temporal**: Plan long-term budget allocation based on predicted trends

### For Researchers
- **GNN**: Explore graph-based spatial analysis methods
- **Temporal**: Study transformer architectures for urban time series

---

## 🤝 Contributing

Both projects are self-contained and can be run independently. For questions or issues:
1. Check the respective project README files
2. Review the dashboard documentation
3. Examine the output files for detailed results

---

## License

This project is part of a datathon submission analyzing accessibility data for urban planning purposes.

---

## Summary

This repository presents two complementary machine learning approaches to accessibility analysis:

1. **GNN Hotspot Detection** identifies **where** high-risk areas are located using graph-based spatial analysis
2. **Temporal Forecasting** predicts **how** accessibility will change over time using transformer-based sequence modeling

Together, they provide complementary analyses of current spatial patterns and projected temporal trends. Both approaches have methodological limitations (GNN: parameter sensitivity, graph construction choices; Temporal: synthetic time proxy, limited validation) that should be considered when interpreting results for infrastructure planning decisions.

**Start exploring:**
- **[GNN Dashboard](./GNN/dashboard.html)** - Interactive hotspot visualization
- **[Temporal Dashboard](./temporal_calc/dashboard.html)** - Interactive forecasting dashboard

## Contact
- Swastik Singh - @swassingh - Swastik.Singh@gmail.com
- Navneeth Dhamotharan - @Navneethd8 - nd17@uw.edu


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[swas-linkedin-shield]: https://img.shields.io/badge/swas-linkedin-blue?style=for-the-badge&logo=linkedin&link=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fswassingh%2F
[swas-linkedin-url]: https://www.linkedin.com/in/swassingh/

[navneeth-linkedin-shield]: https://img.shields.io/badge/navneeth-linkedin-blue?style=for-the-badge&logo=linkedin&link=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fnavneeth-dhamotharan%2F
[navneeth-linkedin-url]: https://www.linkedin.com/in/navneeth-dhamotharan/