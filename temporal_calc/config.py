"""
Configuration module for Transformer Temporal Forecasting Pipeline.
Contains all hyperparameters, paths, and settings.
"""
import os
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
EVAL_DIR = OUTPUT_DIR / "eval"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# Dataset
DATASET_PATH = DATA_DIR / "Access_to_Everyday_Life_Dataset.csv"
PROCESSED_DATA_PATH = OUTPUT_DIR / "processed_timeseries.csv"

# Create directories
for dir_path in [OUTPUT_DIR, PLOTS_DIR, EVAL_DIR, CHECKPOINT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Data Configuration
# ============================================================================
# Number of synthetic time bins to create from attribute_id ordering
# Higher = more granular time series, more training data per neighborhood
NUM_TIME_BINS = 200  # Increased from 50 for denser data

# Columns
COORD_COLS = ["geometry/coordinates/0", "geometry/coordinates/1"]
LABEL_TYPE_COL = "properties/label_type"
NEIGHBORHOOD_COL = "properties/neighborhood"
SEVERITY_COL = "properties/severity"
IS_TEMP_COL = "properties/is_temporary"
ATTRIBUTE_ID_COL = "properties/attribute_id"

# Label types for one-hot encoding
LABEL_TYPES = [
    "CurbRamp", "NoSidewalk", "NoCurbRamp", 
    "SurfaceProblem", "Obstacle", "Occlusion", "Other"
]

# Accessibility Score weights (higher = worse accessibility)
BARRIER_WEIGHTS = {
    "NoCurbRamp": 3.0,
    "NoSidewalk": 3.0,
    "SurfaceProblem": 2.0,
    "Obstacle": 2.5,
    "Occlusion": 1.5,
    "CurbRamp": -1.0,  # Negative because curb ramps IMPROVE accessibility
    "Other": 1.0
}

# ============================================================================
# Model Configuration (Optimized for M1 Mac 8GB RAM)
# ============================================================================
MODEL_CONFIG = {
    "d_model": 128,          # Embedding dimension (increased from 64)
    "n_heads": 8,            # Number of attention heads (increased from 4)
    "n_layers": 3,           # Number of transformer layers (increased from 2)
    "dim_feedforward": 256,  # Feedforward dimension (increased from 128)
    "dropout": 0.15,
    "max_seq_len": 30,       # Maximum sequence length
}

# ============================================================================
# Training Configuration
# ============================================================================
TRAIN_CONFIG = {
    "seq_len": 8,            # Input sequence length (reduced for more samples)
    "pred_len": 1,           # Prediction horizon
    "batch_size": 32,        # Batch size (increased)
    "learning_rate": 5e-4,   # Learning rate (tuned)
    "epochs": 100,
    "patience": 15,          # Early stopping patience (increased)
    "train_ratio": 0.8,      # Train/val split
    "seed": 42,
}

# ============================================================================
# Device Configuration
# ============================================================================
def get_device():
    """Get the best available device (MPS for M1 Mac, else CPU)."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
