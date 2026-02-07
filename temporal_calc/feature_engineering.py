"""
Feature Engineering for Temporal Forecasting.

Creates multivariate time-series features per (time_bin, neighborhood):
- Barrier counts and severity statistics
- Temporary ratio and label type proportions
- Rolling averages and lag features
- Accessibility score as target
"""
import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    DATASET_PATH, OUTPUT_DIR, PROCESSED_DATA_PATH,
    NUM_TIME_BINS, LABEL_TYPE_COL, NEIGHBORHOOD_COL,
    SEVERITY_COL, IS_TEMP_COL, ATTRIBUTE_ID_COL,
    LABEL_TYPES, BARRIER_WEIGHTS
)
from utils import print_section, print_subsection, set_seed


def load_and_prepare_data() -> pd.DataFrame:
    """Load data and create synthetic time bins."""
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # Sort by attribute_id (assumed chronological)
    df = df.sort_values(ATTRIBUTE_ID_COL).reset_index(drop=True)
    
    # Create time bins
    df['time_bin'] = pd.qcut(df.index, q=NUM_TIME_BINS, labels=False)
    
    # Fill missing severity with median
    df[SEVERITY_COL] = df[SEVERITY_COL].fillna(df[SEVERITY_COL].median())
    
    print(f"Loaded {len(df):,} records, {NUM_TIME_BINS} time bins")
    return df


def compute_features_per_group(group: pd.DataFrame) -> pd.Series:
    """
    Compute features for a single (time_bin, neighborhood) group.
    """
    features = {}
    
    # Basic counts
    features['barrier_count'] = len(group)
    
    # Severity statistics
    features['severity_mean'] = group[SEVERITY_COL].mean()
    features['severity_max'] = group[SEVERITY_COL].max()
    features['severity_std'] = group[SEVERITY_COL].std()
    
    # Temporary ratio
    features['temp_ratio'] = group[IS_TEMP_COL].mean()
    
    # Label type proportions
    label_counts = group[LABEL_TYPE_COL].value_counts()
    total = len(group)
    for label in LABEL_TYPES:
        features[f'prop_{label}'] = label_counts.get(label, 0) / total
    
    # Weighted barrier impact (for accessibility score)
    weights = group[LABEL_TYPE_COL].map(BARRIER_WEIGHTS).fillna(1.0)
    features['weighted_impact'] = (weights * group[SEVERITY_COL]).sum()
    
    # Accessibility score (normalized)
    features['accessibility_score'] = (features['weighted_impact'] / features['barrier_count']) * 10
    
    return pd.Series(features)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features and rolling averages.
    """
    print("Adding temporal features (lags and rolling averages)...")
    
    # Sort by neighborhood and time
    df = df.sort_values([NEIGHBORHOOD_COL, 'time_bin']).reset_index(drop=True)
    
    # Features to create lags for
    target_cols = ['barrier_count', 'severity_mean', 'accessibility_score', 'temp_ratio']
    
    # Group by neighborhood to create proper lags
    lag_features = []
    rolling_features = []
    
    for neighborhood in df[NEIGHBORHOOD_COL].unique():
        mask = df[NEIGHBORHOOD_COL] == neighborhood
        neighborhood_df = df[mask].copy()
        
        for col in target_cols:
            # Lag features (t-1, t-2, t-3)
            for lag in [1, 2, 3]:
                neighborhood_df[f'{col}_lag{lag}'] = neighborhood_df[col].shift(lag)
            
            # Rolling averages (window sizes: 3, 5)
            for window in [3, 5]:
                neighborhood_df[f'{col}_roll{window}'] = (
                    neighborhood_df[col].rolling(window=window, min_periods=1).mean()
                )
        
        lag_features.append(neighborhood_df)
    
    df = pd.concat(lag_features, ignore_index=True)
    
    # Fill NaN lags with column mean
    lag_cols = [c for c in df.columns if 'lag' in c or 'roll' in c]
    for col in lag_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time bin position features."""
    print("Adding time position features...")
    
    # Normalize time bin to [0, 1]
    df['time_position'] = df['time_bin'] / df['time_bin'].max()
    
    # Cyclical encoding (for potential periodicity)
    df['time_sin'] = np.sin(2 * np.pi * df['time_position'])
    df['time_cos'] = np.cos(2 * np.pi * df['time_position'])
    
    return df


def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add geospatial density and demographic features (from geospatial_analysis.py).
    """
    print("Adding geospatial and demographic features...")
    
    geo_path = OUTPUT_DIR / 'geospatial_equity_analysis.csv'
    if not geo_path.exists():
        print(f"  Warning: {geo_path} not found, skipping geospatial features")
        return df
    
    geo_df = pd.read_csv(geo_path)
    
    # Select key features to merge
    geo_cols = ['neighborhood', 'barriers_per_km2', 'area_km2']
    demo_cols = ['population', 'median_income', 'pct_minority', 'pct_elderly', 'equity_concern_score']
    
    available_cols = ['neighborhood'] + [c for c in geo_cols[1:] + demo_cols if c in geo_df.columns]
    geo_subset = geo_df[available_cols].copy()
    
    # Normalize numeric features
    for col in available_cols[1:]:
        geo_subset[f'{col}_norm'] = (geo_subset[col] - geo_subset[col].mean()) / geo_subset[col].std()
    
    # Match the neighborhood column name used in the dataframe
    # NEIGHBORHOOD_COL may be 'properties/neighborhood' for CSV data
    nb_col = NEIGHBORHOOD_COL
    if nb_col not in df.columns and 'neighborhood' in df.columns:
        nb_col = 'neighborhood'
    
    # Rename geo column to match
    geo_subset = geo_subset.rename(columns={'neighborhood': nb_col})
    
    # Merge with main dataframe
    df = df.merge(geo_subset, on=nb_col, how='left')
    
    # Fill NaN with 0 for neighborhoods without geo data (0 = mean after normalization)
    norm_cols = [c for c in df.columns if '_norm' in c]
    for col in norm_cols:
        df[col] = df[col].fillna(0)
    
    print(f"  Added {len(norm_cols)} geospatial/demographic features")
    
    return df


def build_features() -> pd.DataFrame:
    """
    Main function to build all features.
    """
    set_seed(42)
    print_section("Feature Engineering Pipeline")
    
    # Load and prepare
    df = load_and_prepare_data()
    
    # Aggregate by (time_bin, neighborhood)
    print_subsection("Computing Features per (time_bin, neighborhood)")
    
    aggregated = df.groupby(['time_bin', NEIGHBORHOOD_COL]).apply(
        compute_features_per_group
    ).reset_index()
    
    print(f"Aggregated to {len(aggregated):,} rows")
    print(f"Shape: {aggregated.shape}")
    
    # Add temporal features
    aggregated = add_temporal_features(aggregated)
    
    # Add time position features
    aggregated = add_time_features(aggregated)
    
    # Add geospatial and demographic features
    aggregated = add_geospatial_features(aggregated)
    
    # Print feature summary
    print_subsection("Feature Summary")
    feature_cols = [c for c in aggregated.columns if c not in ['time_bin', NEIGHBORHOOD_COL]]
    print(f"Total features: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    
    # Save processed data
    print_subsection("Saving Processed Data")
    aggregated.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved to: {PROCESSED_DATA_PATH}")
    
    # Statistics
    print_subsection("Target Variable (accessibility_score) Statistics")
    print(aggregated['accessibility_score'].describe())
    
    return aggregated


def get_feature_columns() -> list:
    """Return list of feature column names (excluding target and identifiers)."""
    exclude = ['time_bin', NEIGHBORHOOD_COL, 'accessibility_score', 'weighted_impact']
    
    # Build feature list
    features = [
        'barrier_count', 'severity_mean', 'severity_max', 'severity_std',
        'temp_ratio', 'time_position', 'time_sin', 'time_cos'
    ]
    
    # Label type proportions
    features += [f'prop_{label}' for label in LABEL_TYPES]
    
    # Lags and rolling averages for key metrics
    for col in ['barrier_count', 'severity_mean', 'accessibility_score', 'temp_ratio']:
        for lag in [1, 2, 3]:
            features.append(f'{col}_lag{lag}')
        for window in [3, 5]:
            features.append(f'{col}_roll{window}')
    
    # Geospatial and demographic features (normalized versions)
    geospatial_features = [
        'barriers_per_km2_norm', 'area_km2_norm',
        'population_norm', 'median_income_norm', 
        'pct_minority_norm', 'pct_elderly_norm',
        'equity_concern_score_norm'
    ]
    features += geospatial_features
    
    return features


if __name__ == "__main__":
    df = build_features()
    print_section("Feature Engineering Complete")
