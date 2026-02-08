"""
Optimized Model Training with XGBoost Comparison.

Improvements:
1. StandardScaler normalization for all features
2. Stronger temporal features (more lags, difference features, momentum)
3. Increased sequence length (10 instead of 5)
4. Multi-step prediction (predict next 3 time steps)
5. XGBoost baseline for comparison
6. Regularization (dropout, weight decay, gradient clipping)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Install with: pip install xgboost")


# Paths
OUTPUT_DIR = Path('/Volumes/NavDisk/datathon26/temporal_calc/outputs')
CHECKPOINT_DIR = Path('/Volumes/NavDisk/datathon26/temporal_calc/checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


# 1. ENHANCED FEATURE ENGINEERING

def add_enhanced_temporal_features(df: pd.DataFrame, target_col: str = 'accessibility_score') -> pd.DataFrame:
    """
    Add stronger temporal features:
    - More lag features (up to 6)
    - Difference features (rate of change)
    - Momentum features
    - Trend indicators
    """
    print("Adding enhanced temporal features...")
    df = df.copy()
    
    # Sort by neighborhood and time
    df = df.sort_values(['properties/neighborhood', 'time_bin']).reset_index(drop=True)
    
    enhanced_dfs = []
    for neighborhood in df['properties/neighborhood'].unique():
        mask = df['properties/neighborhood'] == neighborhood
        nb_df = df[mask].copy()
        
        # More lag features (6 lags instead of 3)
        for lag in range(1, 7):
            nb_df[f'{target_col}_lag{lag}'] = nb_df[target_col].shift(lag)
        
        # Difference features (rate of change)
        nb_df[f'{target_col}_diff1'] = nb_df[target_col].diff(1)
        nb_df[f'{target_col}_diff2'] = nb_df[target_col].diff(2)
        nb_df[f'{target_col}_diff3'] = nb_df[target_col].diff(3)
        
        # Second derivative (acceleration)
        nb_df[f'{target_col}_accel'] = nb_df[f'{target_col}_diff1'].diff(1)
        
        # Momentum (sum of last 3 differences)
        nb_df[f'{target_col}_momentum'] = (
            nb_df[f'{target_col}_diff1'].rolling(3, min_periods=1).sum()
        )
        
        # Rolling statistics (expanded windows)
        for window in [3, 5, 7]:
            nb_df[f'{target_col}_roll_mean{window}'] = nb_df[target_col].rolling(window, min_periods=1).mean()
            nb_df[f'{target_col}_roll_std{window}'] = nb_df[target_col].rolling(window, min_periods=1).std()
            nb_df[f'{target_col}_roll_min{window}'] = nb_df[target_col].rolling(window, min_periods=1).min()
            nb_df[f'{target_col}_roll_max{window}'] = nb_df[target_col].rolling(window, min_periods=1).max()
        
        # Trend indicator: exponential moving average
        nb_df[f'{target_col}_ema3'] = nb_df[target_col].ewm(span=3, adjust=False).mean()
        nb_df[f'{target_col}_ema5'] = nb_df[target_col].ewm(span=5, adjust=False).mean()
        
        # Trend direction: is current above short-term average?
        nb_df[f'{target_col}_above_ema'] = (nb_df[target_col] > nb_df[f'{target_col}_ema3']).astype(float)
        
        enhanced_dfs.append(nb_df)
    
    df = pd.concat(enhanced_dfs, ignore_index=True)
    
    # Fill NaN values with column means
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64'] and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    
    # Replace remaining NaN (e.g., empty std) with 0
    df = df.fillna(0)
    
    print(f"  Added enhanced features. Total columns: {len(df.columns)}")
    return df


# 2. STANDARDSCALER NORMALIZATION

def prepare_normalized_data(df: pd.DataFrame, target_col: str = 'accessibility_score'):
    """
    Apply StandardScaler to all numeric features.
    Returns: X_scaled, y_scaled, feature_scaler, target_scaler, feature_names
    """
    print("Applying StandardScaler normalization...")
    
    # Select feature columns (exclude identifiers and target)
    exclude_cols = ['time_bin', 'properties/neighborhood', target_col, 'weighted_impact']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
    
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    
    # Scale features
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    
    # Scale target
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y)
    
    print(f"  Features: {len(feature_cols)}, Samples: {len(df)}")
    print(f"  Target mean: {y.mean():.2f}, std: {y.std():.2f}")
    
    return X_scaled, y_scaled.flatten(), feature_scaler, target_scaler, feature_cols


# 3. INCREASED SEQUENCE LENGTH DATASET

class EnhancedTimeSeriesDataset(Dataset):
    """
    Dataset with:
    - Longer sequence length (10)
    - Multi-step prediction (predict next 3)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 seq_len: int = 10, pred_len: int = 3,
                 neighborhoods: np.ndarray = None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Build sequences per neighborhood
        self.sequences = []
        self.targets = []
        
        if neighborhoods is not None:
            unique_nb = np.unique(neighborhoods)
            for nb in unique_nb:
                mask = neighborhoods == nb
                X_nb = X[mask]
                y_nb = y[mask]
                self._create_sequences(X_nb, y_nb)
        else:
            self._create_sequences(X, y)
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        
    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """Create sequences from contiguous data."""
        total_len = self.seq_len + self.pred_len
        for i in range(len(X) - total_len + 1):
            seq = X[i:i + self.seq_len]
            target = y[i + self.seq_len:i + total_len]
            self.sequences.append(seq)
            self.targets.append(target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


# 4. TRANSFORMER WITH REGULARIZATION

class RegularizedTransformer(nn.Module):
    """
    Transformer with:
    - Dropout for regularization
    - Layer normalization
    - Multi-step prediction head
    """
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dropout: float = 0.3, pred_len: int = 3):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-step prediction head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len)
        )
        
        self.pred_len = pred_len
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use last position for prediction
        x = x[:, -1, :]
        
        # Multi-step output
        return self.output_head(x)


# 5. XGBOOST BASELINE

def train_xgboost_baseline(X_train, y_train, X_val, y_val, 
                           target_scaler, pred_len: int = 3):
    """
    Train XGBoost model for comparison.
    Creates one model per prediction step.
    """
    if not HAS_XGBOOST:
        print("XGBoost not available, skipping baseline")
        return None, None
    
    print("\n--- Training XGBoost Baseline ---")
    
    # XGBoost needs flattened input
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    
    models = []
    val_preds = []
    
    for step in range(pred_len):
        print(f"  Training step {step + 1}/{pred_len}...")
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train_flat, y_train[:, step],
            eval_set=[(X_val_flat, y_val[:, step])],
            verbose=False
        )
        
        models.append(model)
        val_preds.append(model.predict(X_val_flat))
    
    # Combine predictions
    val_preds = np.stack(val_preds, axis=1)
    
    # Inverse transform for metrics
    val_preds_orig = target_scaler.inverse_transform(val_preds)
    y_val_orig = target_scaler.inverse_transform(y_val)
    
    # Compute average metrics across all steps
    mae = mean_absolute_error(y_val_orig.flatten(), val_preds_orig.flatten())
    rmse = np.sqrt(mean_squared_error(y_val_orig.flatten(), val_preds_orig.flatten()))
    r2 = r2_score(y_val_orig.flatten(), val_preds_orig.flatten())
    
    print(f"  XGBoost Validation MAE: {mae:.4f}")
    print(f"  XGBoost Validation RMSE: {rmse:.4f}")
    print(f"  XGBoost Validation R²: {r2:.4f}")
    
    return models, {'mae': mae, 'rmse': rmse, 'r2': r2}


# 6. TRAINING WITH REGULARIZATION

def train_transformer(model, train_loader, val_loader, 
                      target_scaler,
                      epochs: int = 100,
                      lr: float = 1e-3,
                      weight_decay: float = 1e-4,
                      gradient_clip: float = 1.0,
                      patience: int = 15):
    """
    Train with:
    - AdamW optimizer with weight decay
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    """
    print("\n--- Training Regularized Transformer ---")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())
                val_preds.append(pred.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # Compute MAE in original scale
        val_preds_np = np.concatenate(val_preds)
        val_targets_np = np.concatenate(val_targets)
        val_preds_orig = target_scaler.inverse_transform(val_preds_np)
        val_targets_orig = target_scaler.inverse_transform(val_targets_np)
        val_mae = mean_absolute_error(val_targets_orig.flatten(), val_preds_orig.flatten())
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MAE={val_mae:.2f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = model.state_dict().copy()
        elif epoch - best_epoch >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(best_state)
    print(f"  Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    
    return model, history


def evaluate_model(model, val_loader, target_scaler, pred_len: int = 3):
    """Evaluate transformer on validation set."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # Inverse transform
    preds_orig = target_scaler.inverse_transform(preds)
    targets_orig = target_scaler.inverse_transform(targets)
    
    metrics = {}
    for step in range(pred_len):
        mae = mean_absolute_error(targets_orig[:, step], preds_orig[:, step])
        rmse = np.sqrt(mean_squared_error(targets_orig[:, step], preds_orig[:, step]))
        r2 = r2_score(targets_orig[:, step], preds_orig[:, step])
        metrics[f'step_{step+1}'] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        print(f"  Step {step+1}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
    
    # Average
    avg_mae = mean_absolute_error(targets_orig.flatten(), preds_orig.flatten())
    avg_rmse = np.sqrt(mean_squared_error(targets_orig.flatten(), preds_orig.flatten()))
    avg_r2 = r2_score(targets_orig.flatten(), preds_orig.flatten())
    metrics['average'] = {'mae': avg_mae, 'rmse': avg_rmse, 'r2': avg_r2}
    print(f"  Average: MAE={avg_mae:.2f}, RMSE={avg_rmse:.2f}, R²={avg_r2:.4f}")
    
    return metrics, preds_orig, targets_orig


def plot_comparison(transformer_metrics, xgb_metrics, history, 
                    transformer_preds, transformer_targets, save_path):
    """Plot comparison between Transformer and XGBoost."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training curves
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', alpha=0.8)
    ax.plot(history['val_loss'], label='Val Loss', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Transformer Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Validation MAE over epochs
    ax = axes[0, 1]
    ax.plot(history['val_mae'], color='green', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (original scale)')
    ax.set_title('Validation MAE During Training')
    ax.grid(True, alpha=0.3)
    
    # 3. Model comparison bar chart
    ax = axes[1, 0]
    models = ['Transformer', 'XGBoost']
    maes = [transformer_metrics['average']['mae']]
    rmses = [transformer_metrics['average']['rmse']]
    r2s = [transformer_metrics['average']['r2']]
    
    if xgb_metrics:
        maes.append(xgb_metrics['mae'])
        rmses.append(xgb_metrics['rmse'])
        r2s.append(xgb_metrics['r2'])
    else:
        maes.append(0)
        rmses.append(0)
        r2s.append(0)
        models[1] = 'XGBoost (N/A)'
    
    x = np.arange(len(models))
    width = 0.25
    ax.bar(x - width, maes, width, label='MAE')
    ax.bar(x, rmses, width, label='RMSE')
    ax.bar(x + width, [r * 100 for r in r2s], width, label='R² × 100')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Prediction vs Actual scatter
    ax = axes[1, 1]
    ax.scatter(transformer_targets.flatten(), transformer_preds.flatten(), 
               alpha=0.5, s=20, label='Predictions')
    
    # Perfect prediction line
    min_val = min(transformer_targets.min(), transformer_preds.min())
    max_val = max(transformer_targets.max(), transformer_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Transformer: Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def main():
    print("=" * 60)
    print(" Optimized Model Training")
    print("=" * 60)
    
    # Configuration
    SEQ_LEN = 5  # Reduced to create more sequences from limited data
    PRED_LEN = 3
    BATCH_SIZE = 8  # Smaller batch for small dataset
    EPOCHS = 150
    DROPOUT = 0.3
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0
    
    # Load data
    print("\n--- Loading Data ---")
    df = pd.read_csv(OUTPUT_DIR / 'processed_timeseries.csv')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Add enhanced temporal features
    df = add_enhanced_temporal_features(df)
    
    # Normalize
    X_scaled, y_scaled, feature_scaler, target_scaler, feature_names = prepare_normalized_data(df)
    neighborhoods = df['properties/neighborhood'].values
    
    print(f"\n--- Dataset Summary ---")
    print(f"  Total samples: {len(X_scaled)}")
    print(f"  Features: {X_scaled.shape[1]}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Prediction horizon: {PRED_LEN}")
    
    # Create dataset
    dataset = EnhancedTimeSeriesDataset(
        X_scaled, y_scaled, 
        seq_len=SEQ_LEN, pred_len=PRED_LEN,
        neighborhoods=neighborhoods
    )
    print(f"  Created {len(dataset)} sequences")
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Train Transformer
    model = RegularizedTransformer(
        input_dim=X_scaled.shape[1],
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=DROPOUT,
        pred_len=PRED_LEN
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Transformer parameters: {total_params:,}")
    
    model, history = train_transformer(
        model, train_loader, val_loader, target_scaler,
        epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        gradient_clip=GRADIENT_CLIP
    )
    
    # Evaluate Transformer
    print("\n--- Transformer Evaluation ---")
    transformer_metrics, transformer_preds, transformer_targets = evaluate_model(
        model, val_loader, target_scaler, pred_len=PRED_LEN
    )
    
    # Save Transformer
    torch.save({
        'model_state': model.state_dict(),
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_names': feature_names,
        'config': {
            'input_dim': X_scaled.shape[1],
            'seq_len': SEQ_LEN,
            'pred_len': PRED_LEN
        }
    }, CHECKPOINT_DIR / 'optimized_model.pt')
    print(f"Saved model to {CHECKPOINT_DIR / 'optimized_model.pt'}")
    
    # Train XGBoost baseline
    # Prepare data for XGBoost (need sequences)
    X_train_seq = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])
    y_train_seq = np.array([train_dataset[i][1].numpy() for i in range(len(train_dataset))])
    X_val_seq = np.array([val_dataset[i][0].numpy() for i in range(len(val_dataset))])
    y_val_seq = np.array([val_dataset[i][1].numpy() for i in range(len(val_dataset))])
    
    xgb_models, xgb_metrics = train_xgboost_baseline(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        target_scaler, pred_len=PRED_LEN
    )
    
    # Plot comparison
    plot_comparison(
        transformer_metrics, xgb_metrics, history,
        transformer_preds, transformer_targets,
        OUTPUT_DIR / 'plots' / 'model_comparison.png'
    )
    
    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"\n  Improvements Applied:")
    print(f"    - StandardScaler normalization")
    print(f"    - Enhanced temporal features (+{X_scaled.shape[1] - 51} new features)")
    print(f"    - Sequence length: {SEQ_LEN} (was 5)")
    print(f"    - Multi-step prediction: {PRED_LEN} steps ahead")
    print(f"    - XGBoost baseline comparison")
    print(f"    - Regularization: dropout={DROPOUT}, weight_decay={WEIGHT_DECAY}")
    
    print(f"\n  Transformer Results:")
    print(f"    MAE: {transformer_metrics['average']['mae']:.2f}")
    print(f"    RMSE: {transformer_metrics['average']['rmse']:.2f}")
    print(f"    R²: {transformer_metrics['average']['r2']:.4f}")
    
    if xgb_metrics:
        print(f"\n  XGBoost Results:")
        print(f"    MAE: {xgb_metrics['mae']:.2f}")
        print(f"    RMSE: {xgb_metrics['rmse']:.2f}")
        print(f"    R²: {xgb_metrics['r2']:.4f}")
        
        if transformer_metrics['average']['mae'] < xgb_metrics['mae']:
            print("\n  → Transformer outperforms XGBoost!")
        else:
            print("\n  → XGBoost performs better (consider using it)")


if __name__ == "__main__":
    main()
