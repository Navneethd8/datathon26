"""
PyTorch Dataset Builder for Temporal Forecasting.

Creates sliding window sequences for Transformer input.
Uses global time-series approach (all neighborhoods combined).
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional

from config import (
    PROCESSED_DATA_PATH, NEIGHBORHOOD_COL, TRAIN_CONFIG
)
from feature_engineering import get_feature_columns
from utils import set_seed


class AccessibilityTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for accessibility time-series forecasting.
    
    Uses global time-series approach: treats all neighborhood-time combinations
    as a single sequence, sorted by time_bin then neighborhood.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int = 5,  # Reduced for sparse data
        pred_len: int = 1,
        feature_cols: Optional[List[str]] = None,
        target_col: str = 'accessibility_score',
        normalize: bool = True
    ):
        """
        Args:
            data: DataFrame with time-series features
            seq_len: Length of input sequence
            pred_len: Prediction horizon
            feature_cols: Columns to use as features
            target_col: Column to predict
            normalize: Whether to normalize features
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.feature_cols = feature_cols or get_feature_columns()
        
        # Ensure target is in feature cols for input
        if target_col not in self.feature_cols:
            self.feature_cols = [target_col] + self.feature_cols
        
        # Filter to only existing columns
        available_cols = [c for c in self.feature_cols if c in data.columns]
        self.feature_cols = available_cols
        
        # Sort by time_bin for temporal ordering
        data = data.sort_values('time_bin').reset_index(drop=True)
        
        # Extract feature matrix (global approach - all data as one sequence)
        features = data[self.feature_cols].values.astype(np.float32)
        target_idx = self.feature_cols.index(target_col)
        
        # Build sequences using sliding window on global data
        self.sequences = []
        self.targets = []
        
        for i in range(len(features) - seq_len - pred_len + 1):
            seq = features[i:i + seq_len]
            target = features[i + seq_len:i + seq_len + pred_len, target_idx]
            
            self.sequences.append(seq)
            self.targets.append(target)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
        # Normalize
        self.normalize = normalize
        if normalize and len(self.sequences) > 0:
            self.feature_mean = np.nanmean(self.sequences, axis=(0, 1))
            self.feature_std = np.nanstd(self.sequences, axis=(0, 1)) + 1e-8
            self.sequences = (self.sequences - self.feature_mean) / self.feature_std
            
            # Normalize target with target column stats
            self.target_mean = self.feature_mean[target_idx]
            self.target_std = self.feature_std[target_idx]
            self.targets = (self.targets - self.target_mean) / self.target_std
        else:
            self.feature_mean = None
            self.feature_std = None
            self.target_mean = 0
            self.target_std = 1
        
        # Handle NaN
        self.sequences = np.nan_to_num(self.sequences, nan=0.0)
        self.targets = np.nan_to_num(self.targets, nan=0.0)
        
        print(f"Created dataset: {len(self)} sequences, "
              f"seq_len={seq_len}, pred_len={pred_len}, "
              f"features={len(self.feature_cols)}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.sequences[idx])
        y = torch.from_numpy(self.targets[idx])
        return x, y
    
    def denormalize_target(self, y: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to original scale."""
        if self.normalize:
            return y * self.target_std + self.target_mean
        return y
    
    @property
    def num_features(self) -> int:
        return len(self.feature_cols)


def create_dataloaders(
    data_path: str = None,
    seq_len: int = None,
    pred_len: int = None,
    batch_size: int = None,
    train_ratio: float = None,
    seed: int = None
) -> Tuple[DataLoader, DataLoader, AccessibilityTimeSeriesDataset]:
    """
    Create train and validation DataLoaders.
    
    Returns:
        train_loader, val_loader, full_dataset (for denormalization)
    """
    # Use config defaults
    data_path = data_path or PROCESSED_DATA_PATH
    seq_len = seq_len or TRAIN_CONFIG['seq_len']  # Use config value
    pred_len = pred_len or TRAIN_CONFIG['pred_len']
    batch_size = batch_size or TRAIN_CONFIG['batch_size']
    train_ratio = train_ratio or TRAIN_CONFIG['train_ratio']
    seed = seed or TRAIN_CONFIG['seed']
    
    set_seed(seed)
    
    # Load data
    print(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Create dataset
    dataset = AccessibilityTimeSeriesDataset(
        data=df,
        seq_len=seq_len,
        pred_len=pred_len
    )
    
    # Handle edge case: too few samples
    n_samples = len(dataset)
    if n_samples < 2:
        raise ValueError(f"Not enough samples ({n_samples}) for train/val split. "
                        f"Need more time bins or smaller seq_len.")
    
    n_train = max(1, int(n_samples * train_ratio))
    
    # Use random split
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:] if n_train < n_samples else indices[:1]
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    # Adjust batch size if needed
    actual_batch_size = min(batch_size, len(train_subset))
    
    train_loader = DataLoader(
        train_subset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=0,  # Single thread for M1 compatibility
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")
    
    return train_loader, val_loader, dataset


if __name__ == "__main__":
    # Test dataset creation
    train_loader, val_loader, dataset = create_dataloaders()
    
    # Test batch
    for x, y in train_loader:
        print(f"Batch X shape: {x.shape}")  # [batch, seq_len, features]
        print(f"Batch Y shape: {y.shape}")  # [batch, pred_len]
        break
