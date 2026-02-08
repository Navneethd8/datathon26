"""
Utility to load training history from saved model
"""

import torch
from pathlib import Path
from typing import Dict, Optional


def load_training_history(model_path: str) -> Optional[Dict]:
    """
    Load training history from saved model checkpoint
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Training history dictionary or None if not found
    """
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        return checkpoint.get('history', None)
    except Exception as e:
        print(f"Error loading history: {e}")
        return None

