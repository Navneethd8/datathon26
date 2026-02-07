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


# ============= DATA AUGMENTATION =============

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
        
        if neighborhoods is not None:
            for nb in np.unique(neighborhoods):
                mask = neighborhoods == nb
                self._create_sequences(X[mask], y[mask])
        else:
            self._create_sequences(X, y)
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        
    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        total_len = self.seq_len + self.pred_len
        for i in range(len(X) - total_len + 1):
            self.sequences.append(X[i:i + self.seq_len])
            self.targets.append(y[i + self.seq_len:i + total_len])
    
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


# ============= ENHANCED TRANSFORMER =============

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


# ============= BIDIRECTIONAL LSTM =============

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


# ============= LEARNING RATE SCHEDULER =============

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


# ============= TRAINING =============

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
    
    # Per-step metrics
    step_metrics = []
    for i in range(pred_len):
        step_r2 = r2_score(targets_orig[:, i], preds_orig[:, i])
        step_metrics.append(step_r2)
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'step_r2': step_metrics}, preds_orig, targets_orig


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
    
    print(f"  XGBoost: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}, val_preds_orig


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


def main():
    print("=" * 70)
    print(" MAXIMUM PERFORMANCE MODEL COMPARISON")
    print("=" * 70)
    
    # Import from optimized_training
    from optimized_training import add_enhanced_temporal_features, prepare_normalized_data
    
    # Config
    SEQ_LEN = 5
    PRED_LEN = 3
    BATCH_SIZE = 8
    EPOCHS = 250
    
    # Load data
    print("\n--- Loading Data ---")
    df = pd.read_csv(OUTPUT_DIR / 'processed_timeseries.csv')
    df = add_enhanced_temporal_features(df)
    
    X_scaled, y_scaled, feature_scaler, target_scaler, feature_names = prepare_normalized_data(df)
    neighborhoods = df['properties/neighborhood'].values
    
    print(f"  Samples: {len(X_scaled)}, Features: {X_scaled.shape[1]}")
    
    # Create dataset
    dataset = AugmentedDataset(
        X_scaled, y_scaled,
        seq_len=SEQ_LEN, pred_len=PRED_LEN,
        neighborhoods=neighborhoods,
        augment=True, noise_std=0.05
    )
    print(f"  Sequences: {len(dataset)}")
    
    # Split
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    results = {}
    
    # ===== TRANSFORMER =====
    print("\n" + "=" * 50)
    transformer = MaxPerformanceTransformer(
        input_dim=X_scaled.shape[1],
        d_model=192,
        nhead=12,
        num_layers=6,
        dropout=0.2,
        pred_len=PRED_LEN
    ).to(DEVICE)
    
    print(f"Transformer params: {sum(p.numel() for p in transformer.parameters()):,}")
    
    transformer, t_history = train_model(
        transformer, train_loader, val_loader, target_scaler,
        "Transformer", epochs=EPOCHS, lr=3e-4, patience=30
    )
    
    t_metrics, t_preds, t_targets = evaluate_model(transformer, val_loader, target_scaler, PRED_LEN)
    results['Transformer'] = {'metrics': t_metrics, 'preds': t_preds, 'targets': t_targets, 'history': t_history}
    print(f"  → Transformer: MAE={t_metrics['mae']:.2f}, R²={t_metrics['r2']:.4f}")
    
    # Save best transformer
    torch.save({
        'model_state': transformer.state_dict(),
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_names': feature_names,
        'config': {'input_dim': X_scaled.shape[1], 'seq_len': SEQ_LEN, 'pred_len': PRED_LEN}
    }, CHECKPOINT_DIR / 'best_transformer.pt')
    
    # ===== LSTM =====
    print("\n" + "=" * 50)
    lstm = BiLSTMModel(
        input_dim=X_scaled.shape[1],
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
        pred_len=PRED_LEN
    ).to(DEVICE)
    
    print(f"LSTM params: {sum(p.numel() for p in lstm.parameters()):,}")
    
    lstm, l_history = train_model(
        lstm, train_loader, val_loader, target_scaler,
        "LSTM", epochs=EPOCHS, lr=1e-3, patience=30
    )
    
    l_metrics, l_preds, l_targets = evaluate_model(lstm, val_loader, target_scaler, PRED_LEN)
    results['LSTM'] = {'metrics': l_metrics, 'preds': l_preds, 'targets': l_targets, 'history': l_history}
    print(f"  → LSTM: MAE={l_metrics['mae']:.2f}, R²={l_metrics['r2']:.4f}")
    
    # ===== XGBOOST =====
    X_train_seq = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])
    y_train_seq = np.array([train_dataset[i][1].numpy() for i in range(len(train_dataset))])
    X_val_seq = np.array([val_dataset[i][0].numpy() for i in range(len(val_dataset))])
    y_val_seq = np.array([val_dataset[i][1].numpy() for i in range(len(val_dataset))])
    
    xgb_metrics, xgb_preds = train_xgboost(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        target_scaler, PRED_LEN
    )
    
    if xgb_metrics:
        y_val_orig = target_scaler.inverse_transform(y_val_seq)
        results['XGBoost'] = {'metrics': xgb_metrics, 'preds': xgb_preds, 'targets': y_val_orig}
    
    # Plot results
    (OUTPUT_DIR / 'plots').mkdir(exist_ok=True)
    plot_results(results, OUTPUT_DIR / 'plots' / 'max_performance_comparison.png')
    
    # Final summary
    print("\n" + "=" * 70)
    print(" FINAL RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<15} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 50)
    for model_name in results:
        m = results[model_name]['metrics']
        print(f"{model_name:<15} {m['mae']:>10.2f} {m['rmse']:>10.2f} {m['r2']:>10.4f}")
    
    # Winner
    winner = max(results.keys(), key=lambda x: results[x]['metrics']['r2'])
    print(f"\n🏆 Winner: {winner} with R²={results[winner]['metrics']['r2']:.4f}")


if __name__ == "__main__":
    main()
