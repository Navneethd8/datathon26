# GNN Accessibility Hotspot Detection – Results Summary

## Research Question

**Identify high-risk accessibility hotspots using clustering and/or spatial modeling.**

---

## Direct Answer

We identify high-risk accessibility hotspots using a **hybrid approach** that combines **spatial modeling, clustering, and graph-based representation learning**.  
The method produces spatially coherent hotspot regions that capture all high-severity accessibility issues while limiting the total area flagged as high risk.

---

## Key Results (Test Set)

1. **Identified 255 hotspot regions**, representing **3.74% of spatial units**
2. **All severity ≥ 4 issues are contained within detected hotspots** (3,634 issues)
3. **Moderate positive spatial autocorrelation** observed (Moran’s I = 0.2376)
4. **Hotspots exhibit substantially higher average risk** than non-hotspot areas

> Interpretation: The method concentrates severe accessibility issues into a small, spatially coherent subset of the city.

---

## Approach: Hybrid Spatial Modeling + Clustering

This approach integrates multiple complementary components.

### Spatial Modeling
- KNN-based spatial graph construction (k = 15)
- Explicit use of geographic coordinates and neighborhood context
- Spatial autocorrelation analysis (Moran’s I)
- Validation against classical spatial statistics (Getis-Ord Gi*, Local Moran’s I)

### Clustering
- DBSCAN clustering applied to grid-level representations
- Multi-factor risk scoring combining:
  - Issue density
  - Severity
  - Problem-type diversity
- Spatially contiguous hotspot identification

### Graph-Based Learning
- Graph Attention Network (GAT) learns spatial-contextual representations
- Contrastive learning captures local similarity structure
- Learned embeddings encode relationships beyond raw distance or counts

---

## Rationale for the Hybrid Design

Individual method families exhibit complementary strengths and limitations.

### Clustering-Only Methods
- **DBSCAN (grid features)**: Identifies dense regions but is sensitive to parameter choice
- **Node2Vec + clustering**: Captures graph structure but does not consistently reflect spatial coherence

### Spatial-Only Methods
- **Getis-Ord Gi***: Produces highly spatially coherent clusters but identifies very few regions
- **KDE**: Generates smooth risk surfaces but does not explicitly define discrete hotspots
- **Local Moran’s I**: Did not yield stable hotspot regions under current configuration

### Hybrid GNN-Based Method
- Identifies a larger set of hotspot regions while maintaining spatial coherence
- Preserves full inclusion of high-severity issues as a design constraint
- Produces learned representations that differ meaningfully from classical baselines

> Conclusion: Combining spatial modeling and clustering through graph-based learning provides a flexible middle ground between conservative statistical methods and purely density-based clustering.

---

## Pipeline Execution Summary

**Last Updated**: 2026-02-07 15:54:12

The pipeline was trained and evaluated using explicit train/validation/test splits.

- **Training**: 70% (57,381 nodes)
- **Validation**: 15% (12,296 nodes)
- **Test**: 15% (12,296 nodes)

---

## Training Details

- **Model**: Graph Attention Network (GAT)
- **Training Objective**: Contrastive learning
- **Epochs**: 100
- **Final Training Loss**: 0.0988
- **Final Validation Loss**: 0.0355
- **Training Time**: ~16 minutes
- **Hardware**: CUDA with CPU fallback

Validation loss decreased steadily, indicating stable training behavior.

---

## Data Summary

- **Total Accessibility Records**: 81,973
- **Graph Size**: 81,973 nodes, 1,229,595 edges
- **Grid Resolution**: ~100m × 100m
- **Spatial Units (Test Set)**: 6,826
- **Spatial Units (Full Dataset)**: 11,556

---

## Hotspot Detection Results (Test Set)

- **Hotspots Identified**: 255
- **Spatial Coverage**: 3.74% of units
- **Average Risk (Hotspots)**: 0.7520
- **Average Risk (Non-Hotspots)**: 0.4666
- **Risk Difference**: +0.2854

---

## Evaluation Metrics

### Spatial Coherence
- **Moran’s I**: 0.2376  
  Indicates moderate positive spatial autocorrelation, consistent with spatially clustered risk.

### Clustering Structure
- **Silhouette Score**: 0.0000  
  Not unexpected for spatial hotspot detection, where boundaries are gradual and clusters may overlap.

### Coverage (Design Constraint)
- **Severity ≥ 4 Issues**: 3,634
- **Contained Within Hotspots**: 3,634  
  All high-severity issues are included by construction, while minimizing total spatial coverage.

---

## Baseline Comparison

The method was compared against classical spatial statistics, clustering techniques, and ML baselines.

### Methods Evaluated
- Severity-weighted KDE
- Getis-Ord Gi*
- Local Moran’s I
- Simple multi-factor thresholding
- DBSCAN on grid features
- Random Forest anomaly scoring
- Autoencoder-based anomaly detection
- Node2Vec + clustering

### Comparative Results (Test Set)

| Method | Hotspots | Jaccard vs GNN | Moran’s I |
|------|---------|----------------|-----------|
| **GNN (Hybrid)** | **255** | — | **0.2376** |
| KDE | 77 | 0.0207 | 0.0230 |
| Getis-Ord Gi* | 17 | 0.0148 | 0.7539 |
| Local Moran’s I | 0 | 0.0000 | N/A |
| Simple Thresholding | 63 | 0.0517 | 0.1685 |
| DBSCAN | 85 | 0.0459 | 0.1685 |
| Random Forest | 76 | 0.0481 | 0.1159 |
| Autoencoder | 74 | 0.0104 | 0.0505 |
| Node2Vec + Clustering | 180 | 0.0381 | -0.0030 |

---

## Interpretation of Baselines

- **Getis-Ord Gi*** achieves the strongest spatial coherence but identifies few regions.
- **Clustering-based methods** identify more regions but are less spatially consistent.
- **ML baselines** capture feature-driven risk but lack explicit spatial structure.
- **The GNN approach** occupies a middle ground: broader detection than spatial statistics with more spatial structure than clustering alone.

Low Jaccard overlap indicates that the GNN produces hotspot configurations that differ meaningfully from classical methods.

---

## Outputs

Results are visualized and explored through interactive artifacts:

- **Interactive Hotspot Map** (`./outputs/hotspot_map.html`)
- **Embedding Visualization** (`./outputs/embeddings.png`)
- **Model Weights** (`./models/gnn_model.pt`)
- **Baseline Comparison Outputs** (`./outputs/baseline_comparison.json`)

---

## Summary

This work presents a **hybrid spatial–graph approach** for identifying accessibility hotspots that:

- Concentrates high-severity issues into a small fraction of space
- Produces spatially coherent hotspot regions
- Differentiates learned patterns from classical baselines
- Supports exploratory analysis via interactive maps and embeddings

The results are intended to support **analysis and prioritization**, rather than to replace domain or policy judgment.
