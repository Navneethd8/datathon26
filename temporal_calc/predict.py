"""
Spatial Risk Prediction Module.

Predicts which neighborhoods will have future accessibility problems.
Generates:
- Neighborhood risk rankings
- Risk change predictions
- Geographic risk heatmaps
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    CHECKPOINT_DIR, PLOTS_DIR, PROCESSED_DATA_PATH, OUTPUT_DIR,
    NEIGHBORHOOD_COL, get_device
)
from dataset_builder import AccessibilityTimeSeriesDataset
from model import create_model
from utils import setup_plotting, save_figure, print_section, print_subsection


def load_model_and_data(device: torch.device):
    """Load trained model and processed data."""
    checkpoint_path = CHECKPOINT_DIR / 'best_model.pt'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run train.py first."
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create and load model
    model = create_model(input_dim=checkpoint['num_features'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load processed data
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    return model, checkpoint, df


def predict_neighborhood_risk(model, df, checkpoint, device, seq_len=5):
    """
    Predict future accessibility scores per neighborhood.
    
    Returns DataFrame with:
    - Current accessibility score
    - Predicted future score  
    - Risk change (positive = worsening)
    - Risk level classification
    """
    print_subsection("Predicting Future Accessibility by Neighborhood")
    
    # Get feature columns (same as used in dataset)
    target_col = 'accessibility_score'
    exclude_cols = ['time_bin', NEIGHBORHOOD_COL]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    if target_col not in feature_cols:
        feature_cols = [target_col] + feature_cols
    
    # Normalize using checkpoint stats
    feature_mean = checkpoint['feature_mean']
    feature_std = checkpoint['feature_std']
    target_mean = checkpoint['target_mean']
    target_std = checkpoint['target_std']
    
    # Match the number of features the model expects
    num_model_features = checkpoint['num_features']
    feature_cols = feature_cols[:num_model_features]
    
    predictions = []
    
    for neighborhood in df[NEIGHBORHOOD_COL].unique():
        neighborhood_data = df[df[NEIGHBORHOOD_COL] == neighborhood].sort_values('time_bin')
        
        if len(neighborhood_data) < seq_len:
            continue
        
        # Get last seq_len rows as input
        last_sequence = neighborhood_data[feature_cols].tail(seq_len).values.astype(np.float32)
        
        # Normalize (ensure dimensions match)
        last_sequence = (last_sequence - feature_mean[:num_model_features]) / feature_std[:num_model_features]
        last_sequence = np.nan_to_num(last_sequence, nan=0.0)
        
        # Predict
        x = torch.from_numpy(last_sequence).unsqueeze(0).to(device)  # [1, seq_len, features]
        
        with torch.no_grad():
            pred_normalized = model(x).cpu().numpy().flatten()[0]
        
        # Denormalize
        pred_score = pred_normalized * target_std + target_mean
        
        # Get current score (last available)
        current_score = neighborhood_data['accessibility_score'].iloc[-1]
        
        # Calculate risk change
        risk_change = pred_score - current_score
        
        predictions.append({
            'neighborhood': neighborhood,
            'current_score': current_score,
            'predicted_score': pred_score,
            'risk_change': risk_change,
            'risk_change_pct': (risk_change / (current_score + 1)) * 100
        })
    
    results = pd.DataFrame(predictions)
    
    # Classify risk level
    def classify_risk(row):
        if row['risk_change'] > 5:
            return 'HIGH RISK - Worsening'
        elif row['risk_change'] > 0:
            return 'MODERATE RISK'
        elif row['risk_change'] > -5:
            return 'STABLE'
        else:
            return 'IMPROVING'
    
    results['risk_level'] = results.apply(classify_risk, axis=1)
    results = results.sort_values('risk_change', ascending=False)
    
    return results


def generate_risk_heatmap(risk_results: pd.DataFrame, df: pd.DataFrame):
    """Create visualizations for spatial risk prediction."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top Worsening Neighborhoods
    ax1 = axes[0, 0]
    worsening = risk_results.head(15)
    colors = ['#e74c3c' if x > 5 else '#f39c12' if x > 0 else '#27ae60' 
              for x in worsening['risk_change']]
    bars = ax1.barh(worsening['neighborhood'], worsening['risk_change'], color=colors)
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Predicted Risk Change')
    ax1.set_title('🚨 Top 15 Neighborhoods by Accessibility Risk Change\n(Positive = Worsening)')
    ax1.invert_yaxis()
    
    # 2. Current vs Predicted Score
    ax2 = axes[0, 1]
    ax2.scatter(risk_results['current_score'], risk_results['predicted_score'], 
                c=risk_results['risk_change'], cmap='RdYlGn_r', s=100, alpha=0.7)
    
    # Perfect prediction line
    min_val = min(risk_results['current_score'].min(), risk_results['predicted_score'].min())
    max_val = max(risk_results['current_score'].max(), risk_results['predicted_score'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No Change')
    
    ax2.set_xlabel('Current Accessibility Score')
    ax2.set_ylabel('Predicted Future Score')
    ax2.set_title('Current vs Predicted Accessibility\n(Points above line = worsening)')
    ax2.legend()
    
    # 3. Risk Level Distribution
    ax3 = axes[1, 0]
    risk_counts = risk_results['risk_level'].value_counts()
    risk_colors = {
        'HIGH RISK - Worsening': '#e74c3c',
        'MODERATE RISK': '#f39c12',
        'STABLE': '#3498db',
        'IMPROVING': '#27ae60'
    }
    colors = [risk_colors.get(level, '#95a5a6') for level in risk_counts.index]
    ax3.pie(risk_counts, labels=risk_counts.index, autopct='%1.0f%%', colors=colors, startangle=90)
    ax3.set_title('Neighborhood Risk Level Distribution')
    
    # 4. Top Improving vs Worsening
    ax4 = axes[1, 1]
    top_worsening = risk_results.head(5)[['neighborhood', 'risk_change']].copy()
    top_worsening['type'] = 'Worsening'
    top_improving = risk_results.tail(5)[['neighborhood', 'risk_change']].copy()
    top_improving['type'] = 'Improving'
    
    combined = pd.concat([top_worsening, top_improving])
    colors = ['#e74c3c' if t == 'Worsening' else '#27ae60' for t in combined['type']]
    ax4.barh(combined['neighborhood'], combined['risk_change'], color=colors)
    ax4.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Risk Change')
    ax4.set_title('Top 5 Worsening vs Improving Neighborhoods')
    ax4.invert_yaxis()
    
    plt.suptitle('Spatial Accessibility Risk Prediction', fontsize=16, fontweight='bold')
    save_figure(fig, PLOTS_DIR, 'spatial_risk_prediction')
    
    return fig


def main():
    """Run spatial risk prediction pipeline."""
    setup_plotting()
    print_section("Spatial Risk Prediction")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model and data
    print_subsection("Loading Model")
    model, checkpoint, df = load_model_and_data(device)
    
    # Predict neighborhood risks
    risk_results = predict_neighborhood_risk(model, df, checkpoint, device)
    
    # Print top risk neighborhoods
    print_subsection("🚨 TOP 10 NEIGHBORHOODS AT RISK (Worsening Accessibility)")
    print(risk_results[['neighborhood', 'current_score', 'predicted_score', 'risk_change', 'risk_level']].head(10).to_string(index=False))
    print()
    
    print_subsection("TOP 5 IMPROVING NEIGHBORHOODS")
    print(risk_results[['neighborhood', 'current_score', 'predicted_score', 'risk_change', 'risk_level']].tail(5).to_string(index=False))
    
    # Generate visualizations
    print_subsection("Generating Risk Visualizations")
    generate_risk_heatmap(risk_results, df)
    
    # Save results
    output_path = OUTPUT_DIR / 'neighborhood_risk_forecast.csv'
    risk_results.to_csv(output_path, index=False)
    print(f"\nSaved risk forecast to: {output_path}")
    
    print_section("Spatial Risk Prediction Complete")
    
    return risk_results


if __name__ == "__main__":
    main()
