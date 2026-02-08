# Transformer-Based Temporal Accessibility Forecasting

A lightweight Transformer model for predicting neighborhood accessibility scores based on barrier data.

## Problem Context

This pipeline analyzes the **Access to Everyday Life Dataset** (~82K accessibility barriers across 50 Seattle neighborhoods) to:

1. **Identify barrier patterns** - Which types are most common?
2. **Compare geographic variation** - How do temporary vs permanent barriers distribute?
3. **Predict accessibility** - Forecast future neighborhood accessibility scores

> **Data Constraint**: The dataset lacks timestamps. We create synthetic time bins using `attribute_id` ordering as a temporal proxy.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Input Features (36)                        │
│  barrier_count, severity_mean, label proportions,           │
│  lag features (t-1, t-2, t-3), rolling averages             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Input Projection (36 → 64)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Positional Encoding (Sinusoidal)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Transformer Encoder (2 layers)                  │
│        d_model=64, heads=4, ff=128, dropout=0.1             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Global Average Pooling                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Regression Head → Accessibility Score             │
└─────────────────────────────────────────────────────────────┘
```

**Model Stats**: ~71K parameters, optimized for M1 Mac (8GB RAM)

---

## Quick Start

```bash
# 1. Run EDA (generates plots + accessibility scores)
python eda.py

# 2. Build features
python feature_engineering.py

# 3. Train model
python train.py

# 4. Evaluate
python evaluate.py
```

### Optional Training Arguments

```bash
python train.py --epochs 50 --batch-size 16 --lr 0.001 --patience 10
python train.py --debug  # Quick test (2 epochs)
```

---

## Project Structure

```
temporal_calc/
├── config.py              # Paths, hyperparameters
├── utils.py               # Plotting, early stopping
├── eda.py                 # Exploratory analysis
├── feature_engineering.py # Creates 36-feature time series
├── dataset_builder.py     # PyTorch Dataset (sliding windows)
├── model.py               # Lightweight Transformer
├── train.py               # Training pipeline (MPS/CPU)
├── evaluate.py            # Metrics + visualizations
├── transformer_README.md  # This file
├── outputs/
│   ├── plots/             # EDA visualizations
│   ├── eval/              # Evaluation results
│   ├── accessibility_scores.csv
│   └── processed_timeseries.csv
└── checkpoints/
    └── best_model.pt      # Saved model
```

---

## Key EDA Findings

### Barrier Types (Most Frequent)
| Type | Count | Percentage |
|------|-------|------------|
| CurbRamp | 27,175 | 33.2% |
| NoSidewalk | 19,133 | 23.3% |
| NoCurbRamp | 16,947 | 20.7% |
| SurfaceProblem | 12,681 | 15.5% |

### Worst Accessibility Neighborhoods
1. **Ravenna** (score: 84.8)
2. **Whittier Heights** (score: 83.8)
3. **View Ridge** (score: 81.3)

### Best Accessibility Neighborhoods
1. **First Hill** (score: 7.8)
2. **Belltown** (score: 9.1)
3. **Central Business District** (score: 10.1)

---

## Features (36 total)

| Category | Features |
|----------|----------|
| **Counts** | barrier_count |
| **Severity** | severity_mean, severity_max, severity_std |
| **Ratios** | temp_ratio |
| **Type Proportions** | prop_CurbRamp, prop_NoSidewalk, prop_NoCurbRamp, prop_SurfaceProblem, prop_Obstacle (7 total) |
| **Lag Features** | *_lag1, *_lag2, *_lag3 (for 4 metrics) |
| **Rolling Averages** | *_roll3, *_roll5 (for 4 metrics) |
| **Time Position** | time_position, time_sin, time_cos |

---

## Accessibility Score Formula

```
AccessibilityScore = (Σ barrier_weight × severity) / count × 10
```

**Barrier Weights** (higher = worse for accessibility):
- NoCurbRamp, NoSidewalk: 3.0
- Obstacle: 2.5
- SurfaceProblem: 2.0
- Occlusion: 1.5
- Other: 1.0
- CurbRamp: -1.0 (improves accessibility)

---

## Configuration

Edit `config.py` to adjust:

```python
MODEL_CONFIG = {
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.1,
}

TRAIN_CONFIG = {
    "seq_len": 5,        # Reduced for sparse data
    "batch_size": 16,
    "learning_rate": 1e-3,
    "epochs": 100,
    "patience": 10,
}
```

---

## Technical Notes

- **Device**: Automatically uses MPS (Apple Silicon) if available
- **Memory**: ~280KB model footprint
- **Data Sparsity**: With 50 time bins and 50 neighborhoods, we get ~99 aggregated rows, yielding ~94 training sequences with seq_len=5

---

## Dependencies

```
python >= 3.9
pandas
numpy
matplotlib
seaborn
scikit-learn
torch >= 2.0
```
