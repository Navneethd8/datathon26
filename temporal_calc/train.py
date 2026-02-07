"""
Training Pipeline for Accessibility Transformer.

Features:
- MPS (M1 Mac) and CPU support
- Early stopping
- Checkpoint saving
- MSE/MAE metrics logging
"""
import os
import time
import argparse
import torch
import torch.nn as nn
from pathlib import Path

from config import TRAIN_CONFIG, CHECKPOINT_DIR, get_device
from dataset_builder import create_dataloaders
from model import create_model
from utils import set_seed, EarlyStopping, print_section, print_subsection


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    Train for one epoch.
    
    Returns:
        (avg_loss, avg_mae)
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        pred = model(x)
        loss = criterion(pred, y)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_mae += torch.abs(pred - y).mean().item()
        n_batches += 1
    
    return total_loss / n_batches, total_mae / n_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    Validate model.
    
    Returns:
        (avg_loss, avg_mae, avg_rmse)
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    n_batches = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        pred = model(x)
        loss = criterion(pred, y)
        
        total_loss += loss.item()
        total_mae += torch.abs(pred - y).mean().item()
        total_mse += ((pred - y) ** 2).mean().item()
        n_batches += 1
    
    avg_rmse = (total_mse / n_batches) ** 0.5
    
    return total_loss / n_batches, total_mae / n_batches, avg_rmse


def train(
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    patience: int = None,
    debug: bool = False
):
    """
    Main training function.
    """
    # Use config defaults
    epochs = epochs or TRAIN_CONFIG['epochs']
    batch_size = batch_size or TRAIN_CONFIG['batch_size']
    learning_rate = learning_rate or TRAIN_CONFIG['learning_rate']
    patience = patience or TRAIN_CONFIG['patience']
    
    set_seed(TRAIN_CONFIG['seed'])
    
    print_section("Training Accessibility Transformer")
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # Data
    print_subsection("Loading Data")
    train_loader, val_loader, dataset = create_dataloaders(batch_size=batch_size)
    
    # Model
    print_subsection("Creating Model")
    model = create_model(input_dim=dataset.num_features)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Training loop
    print_subsection("Training")
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': []}
    
    if debug:
        epochs = min(epochs, 2)
        print(f"DEBUG MODE: Running {epochs} epochs")
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_mae, val_rmse = validate(
            model, val_loader, criterion, device
        )
        
        elapsed = time.time() - start_time
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} | "
              f"Val RMSE: {val_rmse:.4f} | "
              f"Time: {elapsed:.1f}s")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'config': TRAIN_CONFIG,
                'num_features': dataset.num_features,
                'feature_mean': dataset.feature_mean,
                'feature_std': dataset.feature_std,
                'target_mean': dataset.target_mean,
                'target_std': dataset.target_std,
            }
            checkpoint_path = CHECKPOINT_DIR / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  -> Saved best model (loss: {val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print_subsection("Training Complete")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR / 'best_model.pt'}")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Accessibility Transformer')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience')
    parser.add_argument('--debug', action='store_true', help='Debug mode (2 epochs)')
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        debug=args.debug
    )
