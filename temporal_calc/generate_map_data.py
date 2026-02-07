"""
Generate block-wise (grid cell) predictions for the interactive map.
Creates a JSON file with barrier data and predictions for each grid cell.
"""
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict

# Paths
DATA_DIR = Path('/Volumes/NavDisk/datathon26/data')
OUTPUT_DIR = Path('/Volumes/NavDisk/datathon26/temporal_calc/outputs')
CHECKPOINT_DIR = Path('/Volumes/NavDisk/datathon26/temporal_calc/checkpoints')

# Grid configuration
GRID_SIZE = 0.005  # ~500m cells at Seattle's latitude


def load_raw_barriers():
    """Load raw barrier data with coordinates."""
    print("Loading raw barrier data...")
    with open(DATA_DIR / 'attributes-seattle.json', 'r') as f:
        data = json.load(f)
    
    barriers = []
    for feature in data['features']:
        props = feature['properties']
        geom = feature['geometry']
        
        if geom['type'] == 'Point':
            coords = geom['coordinates']
            barriers.append({
                'lon': coords[0],
                'lat': coords[1],
                'type': props.get('label_type', 'Unknown'),
                'severity': props.get('severity', 1),
                'is_temporary': props.get('is_temporary', False),
                'neighborhood': props.get('neighborhood', 'Unknown')
            })
    
    print(f"  Loaded {len(barriers)} barriers")
    return pd.DataFrame(barriers)


def create_grid_cells(df: pd.DataFrame):
    """Create grid cells from barrier data."""
    print("Creating grid cells...")
    
    # Get bounds
    min_lat, max_lat = df['lat'].min(), df['lat'].max()
    min_lon, max_lon = df['lon'].min(), df['lon'].max()
    
    print(f"  Bounds: lat [{min_lat:.4f}, {max_lat:.4f}], lon [{min_lon:.4f}, {max_lon:.4f}]")
    
    # Assign each barrier to a grid cell
    df['grid_lat'] = ((df['lat'] - min_lat) / GRID_SIZE).astype(int)
    df['grid_lon'] = ((df['lon'] - min_lon) / GRID_SIZE).astype(int)
    df['cell_id'] = df['grid_lat'].astype(str) + '_' + df['grid_lon'].astype(str)
    
    # Aggregate by cell
    cells = df.groupby('cell_id').agg({
        'lat': 'mean',
        'lon': 'mean',
        'severity': ['count', 'mean', 'max'],
        'is_temporary': 'mean',
        'type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown',
        'neighborhood': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
    }).reset_index()
    
    # Flatten column names
    cells.columns = ['cell_id', 'lat', 'lon', 'barrier_count', 'severity_mean', 
                     'severity_max', 'temp_ratio', 'dominant_type', 'neighborhood']
    
    # Calculate accessibility score per cell
    cells['accessibility_score'] = 100 - (cells['barrier_count'] * cells['severity_mean'] / 10)
    cells['accessibility_score'] = cells['accessibility_score'].clip(0, 100)
    
    # Risk level based on barrier count and severity
    def get_risk(row):
        if row['barrier_count'] >= 50 and row['severity_mean'] >= 3:
            return 'High'
        elif row['barrier_count'] >= 20 or row['severity_mean'] >= 3:
            return 'Medium'
        return 'Low'
    
    cells['risk_level'] = cells.apply(get_risk, axis=1)
    
    # Fill NaN values
    cells = cells.fillna({'severity_mean': 1, 'severity_max': 1, 'temp_ratio': 0, 'accessibility_score': 50})
    
    print(f"  Created {len(cells)} grid cells")
    print(f"  Risk distribution: {cells['risk_level'].value_counts().to_dict()}")
    
    return cells


def load_neighborhood_predictions():
    """Load the neighborhood predictions from the model."""
    pred_path = OUTPUT_DIR / 'neighborhood_risk_forecast.csv'
    if pred_path.exists():
        return pd.read_csv(pred_path)
    return None


def create_map_data(barriers_df: pd.DataFrame, cells_df: pd.DataFrame):
    """Create JSON data for the interactive map."""
    print("Creating map data...")
    
    # Load neighborhood predictions
    nb_preds = load_neighborhood_predictions()
    pred_lookup = {}
    if nb_preds is not None:
        pred_lookup = nb_preds.set_index('neighborhood').to_dict('index')
    
    # Create barrier points (sample for performance)
    sample_size = min(5000, len(barriers_df))
    sample_barriers = barriers_df.sample(n=sample_size, random_state=42)
    
    barrier_points = []
    for _, row in sample_barriers.iterrows():
        severity = row['severity']
        if pd.isna(severity):
            severity = 1
        barrier_points.append({
            'lat': round(row['lat'], 6),
            'lng': round(row['lon'], 6),
            'type': row['type'],
            'severity': int(severity),
            'neighborhood': row['neighborhood']
        })
    
    # Create cell data
    cell_data = []
    for _, row in cells_df.iterrows():
        # Get neighborhood prediction if available
        nb = row['neighborhood']
        pred_info = pred_lookup.get(nb, {})
        
        cell_data.append({
            'lat': round(row['lat'], 6),
            'lng': round(row['lon'], 6),
            'cellId': row['cell_id'],
            'barrierCount': int(row['barrier_count']),
            'severityMean': round(row['severity_mean'], 2),
            'severityMax': int(row['severity_max']),
            'accessibilityScore': round(row['accessibility_score'], 1),
            'dominantType': row['dominant_type'],
            'neighborhood': nb,
            'riskLevel': row['risk_level'],
            'predictedChange': pred_info.get('predicted_change', 0)
        })
    
    # Neighborhood summary
    nb_summary = barriers_df.groupby('neighborhood').agg({
        'lat': 'mean',
        'lon': 'mean',
        'severity': ['count', 'mean']
    }).reset_index()
    nb_summary.columns = ['neighborhood', 'lat', 'lon', 'barrier_count', 'severity_mean']
    
    neighborhood_data = []
    for _, row in nb_summary.iterrows():
        nb = row['neighborhood']
        pred_info = pred_lookup.get(nb, {})
        
        neighborhood_data.append({
            'name': nb,
            'lat': round(row['lat'], 6),
            'lng': round(row['lon'], 6),
            'barrierCount': int(row['barrier_count']),
            'severityMean': round(row['severity_mean'], 2),
            'currentScore': pred_info.get('current_score', 0),
            'predictedScore': pred_info.get('avg_prediction', 0),
            'predictedChange': pred_info.get('predicted_change', 0),
            'riskLevel': pred_info.get('risk_level', 'Unknown')
        })
    
    map_data = {
        'barriers': barrier_points,
        'cells': cell_data,
        'neighborhoods': neighborhood_data,
        'stats': {
            'totalBarriers': len(barriers_df),
            'totalCells': len(cells_df),
            'totalNeighborhoods': len(nb_summary),
            'highRiskCells': len(cells_df[cells_df['risk_level'] == 'High']),
            'mediumRiskCells': len(cells_df[cells_df['risk_level'] == 'Medium']),
            'lowRiskCells': len(cells_df[cells_df['risk_level'] == 'Low'])
        }
    }
    
    return map_data


def main():
    print("=" * 60)
    print(" Block-wise Prediction Generator")
    print("=" * 60)
    
    # Load raw data
    barriers_df = load_raw_barriers()
    
    # Create grid cells
    cells_df = create_grid_cells(barriers_df)
    
    # Create map data
    map_data = create_map_data(barriers_df, cells_df)
    
    # Save JSON for map
    output_path = OUTPUT_DIR / 'map_data.json'
    with open(output_path, 'w') as f:
        json.dump(map_data, f)
    print(f"\nSaved map data to: {output_path}")
    
    # Save cells CSV
    cells_path = OUTPUT_DIR / 'grid_cells.csv'
    cells_df.to_csv(cells_path, index=False)
    print(f"Saved grid cells to: {cells_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"\n  Grid cell size: ~{GRID_SIZE * 111:.0f}m x ~{GRID_SIZE * 85:.0f}m")
    print(f"  Total barriers: {len(barriers_df):,}")
    print(f"  Total grid cells: {len(cells_df):,}")
    print(f"  Barrier sample for map: {len(map_data['barriers']):,}")
    print(f"\n  Risk Distribution:")
    print(f"    High Risk Cells: {map_data['stats']['highRiskCells']}")
    print(f"    Medium Risk Cells: {map_data['stats']['mediumRiskCells']}")
    print(f"    Low Risk Cells: {map_data['stats']['lowRiskCells']}")


if __name__ == "__main__":
    main()
