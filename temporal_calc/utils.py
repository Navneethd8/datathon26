"""
Utility functions for Transformer Temporal Forecasting Pipeline.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_plotting():
    """Configure matplotlib for consistent, publication-quality plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'sans-serif',
    })


def save_figure(fig, path: Path, name: str, tight: bool = True):
    """Save a matplotlib figure to the specified path."""
    if tight:
        fig.tight_layout()
    filepath = path / f"{name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


def print_section(title: str, width: int = 60):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width + "\n")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def format_number(n: float) -> str:
    """Format a number for display."""
    if abs(n) >= 1e6:
        return f"{n/1e6:.1f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.1f}K"
    elif abs(n) >= 1:
        return f"{n:.2f}"
    else:
        return f"{n:.4f}"
