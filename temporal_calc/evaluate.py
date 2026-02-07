"""
Evaluation Module for Accessibility Transformer.

Computes metrics and generates visualizations:
- MAE, RMSE on test set
- Predicted vs Actual scatter plots
- Time-series forecast visualization
- Error distribution analysis
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    CHECKPOINT_DIR, EVAL_DIR, PROCESSED_DATA_PATH,
    get_device, TRAIN_CONFIG
)
from dataset_builder import create_dataloaders
from model import create_model
from utils import setup_plotting, save_figure, print_section, print_subsection


def load_best_model(device: torch.device):
    """Load the best model from checkpoint."""
    checkpoint_path = CHECKPOINT_DIR / 'best_model.pt'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run train.py first."
        )
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = create_model(input_dim=checkpoint['num_features'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(val_loss: {checkpoint['val_loss']:.4f})")
    
    return model, checkpoint


@torch.no_grad()
def evaluate_model(model, val_loader, dataset, device):
    """
    Evaluate model on validation set.
    
    Returns:
        predictions, actuals, metrics
    """
    model.eval()
    
    all_preds = []
    all_actuals = []
    
    for x, y in val_loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        
        all_preds.append(pred)
        all_actuals.append(y.numpy())
    
    preds = np.concatenate(all_preds, axis=0).flatten()
    actuals = np.concatenate(all_actuals, axis=0).flatten()
    
    # Denormalize
    preds_denorm = preds * dataset.target_std + dataset.target_mean
    actuals_denorm = actuals * dataset.target_std + dataset.target_mean
    
    # Compute metrics
    mae = np.abs(preds_denorm - actuals_denorm).mean()
    rmse = np.sqrt(((preds_denorm - actuals_denorm) ** 2).mean())
    mape = np.abs((preds_denorm - actuals_denorm) / (actuals_denorm + 1e-8)).mean() * 100
    
    # Correlation
    correlation = np.corrcoef(preds_denorm, actuals_denorm)[0, 1]
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Correlation': correlation
    }
    
    return preds_denorm, actuals_denorm, metrics


def plot_predictions_vs_actuals(preds, actuals, metrics):
    """Create scatter plot of predictions vs actuals."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(actuals, preds, alpha=0.5, s=30, c='#3498db', edgecolor='white')
    
    # Perfect prediction line
    min_val = min(actuals.min(), preds.min())
    max_val = max(actuals.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Accessibility Score')
    ax.set_ylabel('Predicted Accessibility Score')
    ax.set_title(f"Predictions vs Actuals\n"
                 f"MAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f} | "
                 f"Correlation: {metrics['Correlation']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, EVAL_DIR, 'predictions_vs_actuals')


def plot_error_distribution(preds, actuals, metrics):
    """Plot error distribution."""
    errors = preds - actuals
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(errors, bins=50, color='#e74c3c', edgecolor='white', alpha=0.7)
    ax1.axvline(0, color='black', linestyle='--', lw=2)
    ax1.axvline(errors.mean(), color='blue', linestyle='--', 
                label=f'Mean Error: {errors.mean():.2f}')
    ax1.set_xlabel('Prediction Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')
    ax1.legend()
    
    # Box plot
    ax2 = axes[1]
    ax2.boxplot(errors, vert=True)
    ax2.set_ylabel('Prediction Error')
    ax2.set_title('Error Box Plot')
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    save_figure(fig, EVAL_DIR, 'error_distribution')


def plot_time_series_comparison(preds, actuals, n_samples=100):
    """Plot predicted vs actual as time series."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    n = min(n_samples, len(preds))
    indices = range(n)
    
    ax.plot(indices, actuals[:n], 'b-', label='Actual', linewidth=2, marker='o', 
            markersize=4, alpha=0.7)
    ax.plot(indices, preds[:n], 'r--', label='Predicted', linewidth=2, marker='s',
            markersize=4, alpha=0.7)
    
    ax.fill_between(indices, actuals[:n], preds[:n], alpha=0.2, color='purple')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Accessibility Score')
    ax.set_title(f'Time Series Comparison (first {n} samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, EVAL_DIR, 'time_series_comparison')


def plot_residuals_by_range(preds, actuals):
    """Plot residuals grouped by actual value ranges."""
    errors = preds - actuals
    
    # Create bins
    bins = pd.qcut(actuals, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df = pd.DataFrame({'bin': bins, 'error': errors})
    df.boxplot(column='error', by='bin', ax=ax)
    
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Actual Score Range')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Error by Score Range')
    plt.suptitle('')  # Remove automatic title
    
    save_figure(fig, EVAL_DIR, 'residuals_by_range')


def main():
    """Run full evaluation pipeline."""
    setup_plotting()
    print_section("Model Evaluation")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print_subsection("Loading Model")
    model, checkpoint = load_best_model(device)
    
    # Load data
    print_subsection("Loading Validation Data")
    _, val_loader, dataset = create_dataloaders()
    
    # Evaluate
    print_subsection("Computing Predictions")
    preds, actuals, metrics = evaluate_model(model, val_loader, dataset, device)
    
    # Print metrics
    print_subsection("Metrics")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Generate plots
    print_subsection("Generating Plots")
    plot_predictions_vs_actuals(preds, actuals, metrics)
    plot_error_distribution(preds, actuals, metrics)
    plot_time_series_comparison(preds, actuals)
    plot_residuals_by_range(preds, actuals)
    
    print_section("Evaluation Complete")
    print(f"All plots saved to: {EVAL_DIR}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(EVAL_DIR / 'metrics.csv', index=False)
    print(f"Metrics saved to: {EVAL_DIR / 'metrics.csv'}")
    
    return metrics


if __name__ == "__main__":
    main()
