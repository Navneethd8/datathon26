# Baseline Models Implementation

## Overview

Comprehensive baseline comparison system implemented to compare GNN approach against classical spatial methods, clustering baselines, and ML baselines.

## Implemented Baselines

### A. Classical Spatial Hotspot Baselines 

1. **Severity-weighted KDE**
   - Kernel Density Estimation with severity weights
   - Output: Continuous risk surface → threshold top X% as hotspots
   - File: `src/baseline_models.py::kde_severity_weighted()`

2. **Getis-Ord Gi***
   - Local spatial autocorrelation statistic
   - Identifies statistically significant "hot" cells
   - File: `src/baseline_models.py::getis_ord_gi_star()`

3. **Local Moran's I / LISA**
   - Identifies HH (high-high) clusters
   - Pairs with global Moran's I
   - File: `src/baseline_models.py::local_morans_i()`

4. **Simple Thresholding**
   - Multi-factor scoring: 0.4*density + 0.4*severity + 0.2*diversity
   - No GNN, no DBSCAN - pure feature-based
   - File: `src/baseline_models.py::simple_thresholding()`

### B. Clustering Baselines 

1. **HDBSCAN**
   - More robust than DBSCAN for variable density
   - File: `src/baseline_models.py::hdbscan_clustering()`

2. **OPTICS**
   - Handles variable density better than DBSCAN
   - File: `src/baseline_models.py::optics_clustering()`

3. **DBSCAN Baseline**
   - Standard DBSCAN on grid features
   - File: `src/baseline_models.py::dbscan_clustering()`

4. **KMeans**
   - Non-spatial clustering baseline
   - File: `src/baseline_models.py::kmeans_clustering()`

### C. ML Baselines 

1. **XGBoost/Random Forest**
   - Anomaly score prediction from grid features
   - File: `src/baseline_models.py::xgboost_anomaly_score()`

2. **Autoencoder**
   - Reconstruction error as anomaly score
   - File: `src/baseline_models.py::autoencoder_anomaly_score()`

3. **Node2Vec + Clustering**
   - Graph embedding baseline vs GAT
   - File: `src/baseline_models.py::node2vec_embeddings()`

## Comparison Framework

### Metrics Computed

1. **Jaccard Similarity with GNN**
   - Measures overlap between baseline and GNN hotspots
   - File: `src/baseline_comparison.py::compute_comparison_metrics()`

2. **Spatial Coherence (Moran's I)**
   - Global spatial autocorrelation
   - File: `src/baseline_comparison.py::compute_spatial_coherence()`

3. **Budget vs Coverage** (Framework ready)
   - For top 1%, 2%, 5%, 10% of area
   - Coverage of high-severity problems
   - File: `src/baseline_comparison.py::budget_coverage_curve()`

4. **Stability Analysis** (Framework ready)
   - Jaccard overlap across runs
   - File: `src/baseline_comparison.py::compute_stability()`

## Integration

### Pipeline Integration
- **Step 11**: Baseline Comparison (runs automatically with `--eval`)
- Location: `src/main.py` lines 221-266

### Output Files
- `outputs/baseline_comparison.json` - All comparison results
- `outputs/baseline_comparison/comparison_table.png` - Visual comparison table
- `outputs/baseline_comparison/budget_coverage.png` - Budget vs coverage plot
- `RESULTS_SUMMARY.md` - Updated with baseline comparison section

### Automatic Updates
- `RESULTS_SUMMARY.md` automatically updated after baseline comparison
- Comparison table with all methods
- Key findings and insights

## Usage

### Run Full Pipeline with Baselines
```bash
python -m src.main --train --eval --visualize
```

This will:
1. Train GNN model
2. Run all baseline methods
3. Compute comparison metrics
4. Generate visualizations
5. Update RESULTS_SUMMARY.md

### Run Baselines Only (if model exists)
```bash
python -m src.main --eval --visualize
```

### Update Summary Manually
```bash
python -m src.update_results_with_baselines
```

## Dependencies Added

- `hdbscan>=0.8.0` - HDBSCAN clustering
- `node2vec>=0.4.0` - Node2Vec embeddings
- `networkx>=3.0` - Graph processing for Node2Vec

## Files Created

1. `src/baseline_models.py` - All baseline method implementations
2. `src/baseline_comparison.py` - Comparison framework
3. `src/baseline_visualization.py` - Comparison visualizations
4. `src/update_results_with_baselines.py` - Auto-update RESULTS_SUMMARY.md

## Comparison Table Format

The RESULTS_SUMMARY.md will include:

| Method | Hotspots Detected | Jaccard with GNN | Spatial Coherence (Moran's I) |
|--------|-------------------|------------------|-------------------------------|
| GNN (Our Method) | X | 1.000 | X.XXXX |
| KDE (Severity-weighted) | X | X.XXXX | X.XXXX |
| Getis-Ord Gi* | X | X.XXXX | X.XXXX |
| ... | ... | ... | ... |

## Next Steps

1. **Run the pipeline** to generate baseline comparisons
2. **Review comparison results** in `outputs/baseline_comparison.json`
3. **Check updated RESULTS_SUMMARY.md** for comparison section
4. **Analyze visualizations** in `outputs/baseline_comparison/`

## Notes

- Some baselines require optional dependencies (hdbscan, node2vec)
- If dependencies are missing, those baselines will be skipped with warnings
- All results are saved to JSON for further analysis
- Comparison metrics are automatically computed and added to summary

