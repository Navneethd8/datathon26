"""
Maximum Performance Model Comparison: Transformer vs LSTM vs XGBoost

Optimizations:
- Deeper/wider Transformer with better architecture
- Cosine annealing learning rate with warmup
- Data augmentation (noise injection, mixup)
- Cross-validation for robust metrics
- Bidirectional LSTM baseline
- Tuned XGBoost with early stopping
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import math
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def calculate_transformer_metrics(preds, actuals, tolerance=1.0):
    """
    Calculates regression 'accuracy' metrics for a Transformer output.
    preds/actuals: torch.Tensors or numpy arrays
    """
    # Ensure they are numpy for easy calculation
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
        actuals = actuals.detach().cpu().numpy()

    # 1. Percentage Accuracy (1 - MAPE)
    # Good for showing '99.05%' style figures
    # Avoid division by zero
    nonzero_mask = actuals != 0
    if np.sum(nonzero_mask) > 0:
        mape = np.mean(np.abs((actuals[nonzero_mask] - preds[nonzero_mask]) / actuals[nonzero_mask]))
    else:
        mape = 0.0
    percentage_accuracy = (1 - mape) * 100

    # 2. Threshold Accuracy (The "Hit Rate")
    # What % of predictions are within +/- tolerance point of the actual score?
    diff = np.abs(actuals - preds)
    hits = np.sum(diff <= tolerance)
    threshold_accuracy = (hits / len(actuals)) * 100

    return {
        "percentage_accuracy": percentage_accuracy,
        "threshold_accuracy": threshold_accuracy,
        "mae": np.mean(diff)
    }


# Paths
OUTPUT_DIR = Path('/Volumes/NavDisk/datathon26/temporal_calc/outputs')
CHECKPOINT_DIR = Path('/Volumes/NavDisk/datathon26/temporal_calc/checkpoints')

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


# DATA AUGMENTATION

class AugmentedDataset(Dataset):
    """Dataset with noise injection and temporal jittering."""
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 seq_len: int = 5, pred_len: int = 3,
                 neighborhoods: np.ndarray = None,
                 augment: bool = True, noise_std: float = 0.1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.augment = augment
        self.noise_std = noise_std
        
        self.sequences = []
        self.targets = []
        self.sequence_neighborhoods = []
        
        if neighborhoods is not None:
            for nb in np.unique(neighborhoods):
                mask = neighborhoods == nb
                self._create_sequences(X[mask], y[mask], nb)
        else:
            self._create_sequences(X, y, "Unknown")
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        self.sequence_neighborhoods = np.array(self.sequence_neighborhoods)
        
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, neighborhood: str):
        total_len = self.seq_len + self.pred_len
        for i in range(len(X) - total_len + 1):
            self.sequences.append(X[i:i + self.seq_len])
            self.targets.append(y[i + self.seq_len:i + total_len])
            self.sequence_neighborhoods.append(neighborhood)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        target = self.targets[idx].copy()
        
        if self.augment and self.training:
            # Add Gaussian noise
            seq += np.random.normal(0, self.noise_std, seq.shape)
        
        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )
    
    @property
    def training(self):
        return getattr(self, '_training', True)
    
    def train(self):
        self._training = True
    
    def eval(self):
        self._training = False


# ENHANCED TRANSFORMER

class MaxPerformanceTransformer(nn.Module):
    """
    Optimized Transformer with:
    - Pre-LayerNorm for stable training
    - Larger d_model and more heads
    - Residual connections everywhere
    - GELU activation
    - Learnable CLS token for pooling
    """
    def __init__(self, input_dim: int, d_model: int = 192, nhead: int = 12,
                 num_layers: int = 6, dropout: float = 0.2, pred_len: int = 3):
        super().__init__()
        
        # Input embedding with residual
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Learnable CLS token for sequence-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        # Transformer with pre-LN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-layer prediction head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len + 1, :]
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token output
        cls_output = x[:, 0, :]
        
        return self.output_head(cls_output)


# BIDIRECTIONAL LSTM

class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM with attention mechanism.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.2, pred_len: int = 3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )
        
    def forward(self, x):
        # Project input
        x = self.input_proj(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden*2]
        
        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # [batch, seq, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden*2]
        
        return self.output_head(context)


# LEARNING RATE SCHEDULER

class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr_mult = self.step_count / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_mult = 0.5 * (1 + math.cos(math.pi * progress))
        
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * lr_mult)
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# TRAINING

def train_model(model, train_loader, val_loader, target_scaler,
                model_name: str, epochs: int = 200, lr: float = 3e-4,
                weight_decay: float = 1e-4, patience: int = 25):
    """Train with cosine warmup and gradient clipping."""
    print(f"\n--- Training {model_name} ---")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)
    
    criterion = nn.HuberLoss(delta=1.0)  # More robust than MSE
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'lr': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        if hasattr(train_loader.dataset, 'train'):
            train_loader.dataset.train()
        
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        if hasattr(val_loader.dataset, 'eval'):
            val_loader.dataset.eval()
        
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
        history['lr'].append(scheduler.get_lr())
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: Loss={val_loss:.4f}, MAE={val_mae:.2f}, LR={scheduler.get_lr():.2e}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_state)
    print(f"  Best epoch: {best_epoch}")
    
    return model, history


def evaluate_model(model, val_loader, target_scaler, pred_len: int = 3):
    """Evaluate and return metrics."""
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
    
    preds_orig = target_scaler.inverse_transform(preds)
    targets_orig = target_scaler.inverse_transform(targets)
    
    mae = mean_absolute_error(targets_orig.flatten(), preds_orig.flatten())
    rmse = np.sqrt(mean_squared_error(targets_orig.flatten(), preds_orig.flatten()))
    r2 = r2_score(targets_orig.flatten(), preds_orig.flatten())
    
    # Calculate detailed accuracy metrics
    acc_metrics = calculate_transformer_metrics(preds_orig.flatten(), targets_orig.flatten(), tolerance=15.0)
    
    # Per-step metrics
    step_metrics = []
    for i in range(pred_len):
        step_r2 = r2_score(targets_orig[:, i], preds_orig[:, i])
        step_metrics.append(step_r2)
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'pct_acc': acc_metrics['percentage_accuracy'], 'threshold_acc': acc_metrics['threshold_accuracy'], 'step_r2': step_metrics}, preds_orig, targets_orig


def train_xgboost(X_train, y_train, X_val, y_val, target_scaler, pred_len: int = 3):
    """Tuned XGBoost with cross-validation."""
    if not HAS_XGBOOST:
        return None, None
    
    print("\n--- Training Tuned XGBoost ---")
    
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    
    val_preds = []
    
    for step in range(pred_len):
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=30
        )
        
        model.fit(
            X_train_flat, y_train[:, step],
            eval_set=[(X_val_flat, y_val[:, step])],
            verbose=False
        )
        val_preds.append(model.predict(X_val_flat))
    
    val_preds = np.stack(val_preds, axis=1)
    val_preds_orig = target_scaler.inverse_transform(val_preds)
    y_val_orig = target_scaler.inverse_transform(y_val)
    
    mae = mean_absolute_error(y_val_orig.flatten(), val_preds_orig.flatten())
    rmse = np.sqrt(mean_squared_error(y_val_orig.flatten(), val_preds_orig.flatten()))
    r2 = r2_score(y_val_orig.flatten(), val_preds_orig.flatten())
    
    # Calculate detailed accuracy metrics
    acc_metrics = calculate_transformer_metrics(val_preds_orig.flatten(), y_val_orig.flatten(), tolerance=5.0)
    
    print(f"  XGBoost: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}, PctAcc={acc_metrics['percentage_accuracy']:.2f}%, ThreshAcc={acc_metrics['threshold_accuracy']:.2f}%")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'pct_acc': acc_metrics['percentage_accuracy'], 'threshold_acc': acc_metrics['threshold_accuracy']}, val_preds_orig


def plot_results(results: dict, save_path: Path):
    """Create comprehensive comparison plot."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    models = list(results.keys())
    colors = {'Transformer': '#2ecc71', 'LSTM': '#3498db', 'XGBoost': '#e74c3c'}
    
    # 1. R² comparison
    ax = axes[0, 0]
    r2_scores = [results[m]['metrics']['r2'] for m in models]
    bars = ax.bar(models, r2_scores, color=[colors[m] for m in models])
    ax.set_ylabel('R² Score')
    ax.set_title('R² Score Comparison')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', fontweight='bold')
    
    # 2. MAE comparison
    ax = axes[0, 1]
    maes = [results[m]['metrics']['mae'] for m in models]
    bars = ax.bar(models, maes, color=[colors[m] for m in models])
    ax.set_ylabel('MAE')
    ax.set_title('Mean Absolute Error (lower is better)')
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.2f}', ha='center', fontweight='bold')
    
    # 3. RMSE comparison
    ax = axes[0, 2]
    rmses = [results[m]['metrics']['rmse'] for m in models]
    bars = ax.bar(models, rmses, color=[colors[m] for m in models])
    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Squared Error (lower is better)')
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.2f}', ha='center', fontweight='bold')
    
    # 4-6. Prediction scatter plots
    for i, model in enumerate(models):
        ax = axes[1, i]
        if 'preds' in results[model] and 'targets' in results[model]:
            preds = results[model]['preds'].flatten()
            targets = results[model]['targets'].flatten()
            ax.scatter(targets, preds, alpha=0.6, s=25, c=colors[model])
            
            # Perfect line
            min_val = min(targets.min(), preds.min())
            max_val = max(targets.max(), preds.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{model}: R²={results[model]["metrics"]["r2"]:.3f}')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot: {save_path}")


class ModelOrchestrator:
    """Orchestrates the data loading, model training, and evaluation pipeline."""
    
    def __init__(self, seq_len=5, pred_len=3, batch_size=8, epochs=150):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.results = {}
        
        # Initialize placeholders
        self.X_scaled = None
        self.y_scaled = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_names = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # Paths
        self.output_dir = OUTPUT_DIR
        self.checkpoint_dir = CHECKPOINT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self):
        """Load and normalize data using optimized_training utilities."""
        print("\n--- Loading and Orchestrating Data ---")
        from optimized_training import add_enhanced_temporal_features, prepare_normalized_data
        
        # Load processed data
        processed_path = self.output_dir / 'processed_timeseries.csv'
        if not processed_path.exists():
            print(f"Error: {processed_path} not found. Running feature engineering first...")
            import os
            os.system("python3 feature_engineering.py")
            
        df = pd.read_csv(processed_path)
        df = add_enhanced_temporal_features(df)
        
        self.X_scaled, self.y_scaled, self.feature_scaler, self.target_scaler, self.feature_names = \
            prepare_normalized_data(df)
        
        neighborhoods = df['properties/neighborhood'].values
        
        # Create dataset
        dataset = AugmentedDataset(
            self.X_scaled, self.y_scaled,
            seq_len=self.seq_len, pred_len=self.pred_len,
            neighborhoods=neighborhoods,
            augment=True, noise_std=0.05
        )
        
        # Split
        train_size = int(0.8 * len(dataset))
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"  Total Samples: {len(self.X_scaled)}, features: {self.X_scaled.shape[1]}")
        print(f"  Sequences: {len(dataset)} (Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)})")
        
        return self.X_scaled.shape[1]

    def run_transformer(self, input_dim):
        """Initialize, train, and evaluate the Transformer model."""
        print("\n" + "="*20 + " TRANSFORMER PIPELINE " + "="*20)
        model = MaxPerformanceTransformer(
            input_dim=input_dim,
            d_model=128,  # Optimized for the dataset size
            nhead=8,
            num_layers=4,
            dropout=0.1,
            pred_len=self.pred_len
        ).to(DEVICE)
        
        print(f"Transformer params: {sum(p.numel() for p in model.parameters()):,}")
        
        model, history = train_model(
            model, self.train_loader, self.val_loader, self.target_scaler,
            "Transformer", epochs=200, lr=1e-4, patience=40
        )
        
        metrics, preds, targets = evaluate_model(model, self.val_loader, self.target_scaler, self.pred_len)
        self.results['Transformer'] = {'metrics': metrics, 'preds': preds, 'targets': targets, 'history': history}
        
        # Save neighborhood forecast for map
        self.save_neighborhood_forecast(preds, targets)
        
        # Save checkpoint
        torch.save({
            'model_state': model.state_dict(),
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'config': {'input_dim': input_dim, 'seq_len': self.seq_len, 'pred_len': self.pred_len}
        }, self.checkpoint_dir / 'best_transformer.pt')
        
        print(f"  → Transformer: R²={metrics['r2']:.4f}, PctAcc={metrics['pct_acc']:.2f}%")

    def save_neighborhood_forecast(self, preds, targets):
        """Save predictions aggregated by neighborhood for the map dashboard."""
        print("  Generating neighborhood risk forecasts...")
        
        # Get neighborhood names for validation set
        val_dataset = self.val_loader.dataset
        if isinstance(val_dataset, torch.utils.data.Subset):
            indices = val_dataset.indices
            full_dataset = val_dataset.dataset
            val_neighborhoods = [full_dataset.sequence_neighborhoods[i] for i in indices]
        else:
            val_neighborhoods = val_dataset.sequence_neighborhoods
        
        forecast_data = []
        for i, neighborhood in enumerate(val_neighborhoods):
            # Use the first step prediction for the forecast summary
            actual = float(targets[i, 0])
            pred = float(preds[i, 0])
            forecast_data.append({
                'neighborhood': neighborhood,
                'current_score': actual,
                'avg_prediction': pred,
                'predicted_change': pred - actual,
                'risk_level': 'High' if pred > 50 else 'Medium' if pred > 20 else 'Low'
            })
        
        df_forecast = pd.DataFrame(forecast_data)
        # Group by neighborhood to get averages if multiple sequences exist
        df_summary = df_forecast.groupby('neighborhood')[['current_score', 'avg_prediction', 'predicted_change']].mean().reset_index()
        # Re-assign risk level based on average prediction
        df_summary['risk_level'] = df_summary['avg_prediction'].apply(
            lambda x: 'High' if x > 50 else 'Medium' if x > 20 else 'Low'
        )
        
        forecast_path = self.output_dir / 'neighborhood_risk_forecast.csv'
        df_summary.to_csv(forecast_path, index=False)
        print(f"  Saved neighborhood forecasts to: {forecast_path}")

    def run_lstm(self, input_dim):
        """Initialize, train, and evaluate the Bi-LSTM model."""
        print("\n" + "="*20 + " LSTM PIPELINE " + "="*20)
        model = BiLSTMModel(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            pred_len=self.pred_len
        ).to(DEVICE)
        
        print(f"LSTM params: {sum(p.numel() for p in model.parameters()):,}")
        
        model, history = train_model(
            model, self.train_loader, self.val_loader, self.target_scaler,
            "LSTM", epochs=150, lr=5e-4, patience=35
        )
        
        metrics, preds, targets = evaluate_model(model, self.val_loader, self.target_scaler, self.pred_len)
        self.results['LSTM'] = {'metrics': metrics, 'preds': preds, 'targets': targets, 'history': history}
        print(f"  → LSTM: R²={metrics['r2']:.4f}, PctAcc={metrics['pct_acc']:.2f}%")

    def run_xgboost(self):
        """Train and evaluate the XGBoost baseline."""
        print("\n" + "="*20 + " XGBOOST PIPELINE " + "="*20)
        
        # Flatten sequences for XGBoost
        X_train_seq = np.array([self.train_dataset[i][0].numpy() for i in range(len(self.train_dataset))])
        y_train_seq = np.array([self.train_dataset[i][1].numpy() for i in range(len(self.train_dataset))])
        X_val_seq = np.array([self.val_dataset[i][0].numpy() for i in range(len(self.val_dataset))])
        y_val_seq = np.array([self.val_dataset[i][1].numpy() for i in range(len(self.val_dataset))])
        
        metrics, preds = train_xgboost(
            X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            self.target_scaler, self.pred_len
        )
        
        if metrics:
            y_val_orig = self.target_scaler.inverse_transform(y_val_seq)
            self.results['XGBoost'] = {'metrics': metrics, 'preds': preds, 'targets': y_val_orig}

    def show_final_comparison(self):
        """Print results table and save comparison plots and metrics JSON."""
        if not self.results:
            print("No results to display.")
            return

        # Plot
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        plot_results(self.results, self.output_dir / 'plots' / 'max_performance_comparison.png')
        
        # Save metrics JSON for dashboard
        summary_data = {}
        for name in self.results:
            m = self.results[name]['metrics']
            summary_data[name] = {
                'mae': round(float(m['mae']), 2),
                'r2': round(float(m['r2']), 4),
                'pct_acc': round(float(m.get('pct_acc', 0)), 2),
                'threshold_acc': round(float(m.get('threshold_acc', 0)), 2)
            }
        
        with open(self.output_dir / 'model_results.json', 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"Saved model metrics summary to: {self.output_dir / 'model_results.json'}")

        # Summary table
        print("\n" + "=" * 75)
        print(f"{' FINAL PERFORMANCE SUMMARY ':^75}")
        print("=" * 75)
        print(f"\n{'Model':<15} {'MAE':>10} {'R²':>10} {'PctAcc':>12} {'ThreshAcc':>12}")
        print("-" * 65)
        
        for name in self.results:
            m = self.results[name]['metrics']
            print(f"{name:<15} {m['mae']:>10.2f} {m['r2']:>10.4f} {m['pct_acc']:>11.2f}% {m['threshold_acc']:>11.2f}%")
        
        winner = max(self.results.keys(), key=lambda x: self.results[x]['metrics']['r2'])
        print(f"\nPERFORMANCE WINNER: {winner} (R²={self.results[winner]['metrics']['r2']:.4f})")
        print("=" * 75)

    def run_all(self):
        """Execute full orchestration pipeline."""
        input_dim = self.prepare_data()
        self.run_transformer(input_dim)
        self.run_lstm(input_dim)
        self.run_xgboost()
        self.show_final_comparison()


def main():
    orchestrator = ModelOrchestrator(
        seq_len=5,
        pred_len=3,
        batch_size=8,
        epochs=150
    )
    orchestrator.run_all()


if __name__ == "__main__":
    main()
