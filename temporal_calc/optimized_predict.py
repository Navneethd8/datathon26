"""
Generate neighborhood risk forecasts using the optimized model.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import from optimized_training
from optimized_training import (
    RegularizedTransformer,
    add_enhanced_temporal_features,
    DEVICE, OUTPUT_DIR, CHECKPOINT_DIR
)

def load_optimized_model():
    """Load the trained optimized model with scalers."""
    checkpoint = torch.load(CHECKPOINT_DIR / 'optimized_model.pt', weights_only=False)
    
    config = checkpoint['config']
    model = RegularizedTransformer(
        input_dim=config['input_dim'],
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.0,  # No dropout for inference
        pred_len=config['pred_len']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, checkpoint['feature_scaler'], checkpoint['target_scaler'], config


def prepare_features(df: pd.DataFrame, feature_scaler, feature_names: list) -> np.ndarray:
    """Prepare and scale features for prediction."""
    # Ensure all feature columns exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # Fill missing with 0
    
    X = df[feature_names].values
    X_scaled = feature_scaler.transform(X)
    return X_scaled


def predict_neighborhood_risk():
    """Generate forecasts for each neighborhood."""
    print("=" * 60)
    print(" Neighborhood Risk Prediction (Optimized Model)")
    print("=" * 60)
    
    # Load model
    print("\nLoading optimized model...")
    model, feature_scaler, target_scaler, config = load_optimized_model()
    print(f"  Model loaded: {config['input_dim']} features, seq_len={config['seq_len']}, pred_len={config['pred_len']}")
    
    # Load and prepare data
    print("\nPreparing data...")
    df = pd.read_csv(OUTPUT_DIR / 'processed_timeseries.csv')
    df = add_enhanced_temporal_features(df)
    
    # Get neighborhoods
    nb_col = 'properties/neighborhood'
    neighborhoods = df[nb_col].unique()
    print(f"  Found {len(neighborhoods)} neighborhoods")
    
    # Get feature names from scaler
    feature_names = list(feature_scaler.feature_names_in_) if hasattr(feature_scaler, 'feature_names_in_') else None
    
    if feature_names is None:
        # Fallback: extract from config
        exclude_cols = ['time_bin', nb_col, 'accessibility_score', 'weighted_impact']
        feature_names = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
    
    results = []
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for nb in neighborhoods:
            nb_df = df[df[nb_col] == nb].sort_values('time_bin')
            
            if len(nb_df) < config['seq_len']:
                print(f"  Skipping {nb}: insufficient data ({len(nb_df)} < {config['seq_len']})")
                continue
            
            # Get last seq_len rows
            recent = nb_df.tail(config['seq_len']).copy()
            
            # Prepare features
            X_scaled = prepare_features(recent, feature_scaler, feature_names)
            
            # Get current score (last value)
            current_score = recent['accessibility_score'].iloc[-1]
            
            # Create sequence tensor [1, seq_len, features]
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Predict
            pred_scaled = model(X_tensor).cpu().numpy()  # [1, pred_len]
            pred_original = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            avg_prediction = pred_original.mean()
            predicted_change = avg_prediction - current_score
            risk_level = 'High' if predicted_change < -10 else ('Medium' if predicted_change < -5 else 'Low')
            
            results.append({
                'neighborhood': nb,
                'current_score': round(current_score, 2),
                'predicted_step1': round(pred_original[0], 2),
                'predicted_step2': round(pred_original[1], 2),
                'predicted_step3': round(pred_original[2], 2),
                'avg_prediction': round(avg_prediction, 2),
                'predicted_change': round(predicted_change, 2),
                'risk_level': risk_level
            })
    
    # Create DataFrame and sort by risk
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('predicted_change')
    
    # Save
    output_path = OUTPUT_DIR / 'neighborhood_risk_forecast.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print(" FORECAST SUMMARY")
    print("=" * 60)
    print(f"\nNeighborhoods by Risk Level:")
    print(f"  High Risk: {len(results_df[results_df['risk_level'] == 'High'])}")
    print(f"  Medium Risk: {len(results_df[results_df['risk_level'] == 'Medium'])}")
    print(f"  Low Risk: {len(results_df[results_df['risk_level'] == 'Low'])}")
    
    print(f"\nTop 5 At-Risk Neighborhoods:")
    for _, row in results_df.head(5).iterrows():
        print(f"  {row['neighborhood']}: {row['current_score']:.1f} → {row['avg_prediction']:.1f} ({row['predicted_change']:+.1f})")
    
    print(f"\nMost Stable/Improving Neighborhoods:")
    for _, row in results_df.tail(5).iterrows():
        print(f"  {row['neighborhood']}: {row['current_score']:.1f} → {row['avg_prediction']:.1f} ({row['predicted_change']:+.1f})")
    
    return results_df


if __name__ == "__main__":
    predict_neighborhood_risk()
