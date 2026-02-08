"""
Generate map data for GNN dashboard
Creates a JSON file with hotspot data, grid cells, and statistics for the interactive dashboard.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# Paths
OUTPUT_DIR = Path(__file__).parent / 'outputs'
DATA_DIR = Path(__file__).parent.parent / 'data'


def load_gnn_results() -> Dict:
    """Load GNN results from JSON files"""
    results_path = OUTPUT_DIR / 'results.json'
    baseline_path = OUTPUT_DIR / 'baseline_comparison.json'
    
    results = {}
    if results_path.exists():
        with open(results_path, 'r') as f:
            results['evaluation'] = json.load(f)
    
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            results['baseline'] = json.load(f)
    
    return results


def load_raw_data() -> pd.DataFrame:
    """Load raw accessibility data"""
    csv_path = DATA_DIR / 'Access_to_Everyday_Life_Dataset.csv'
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Using empty dataframe.")
        return pd.DataFrame()
    
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract coordinates
    lon_col = None
    lat_col = None
    for col in df.columns:
        if 'coordinates/0' in col or col == 'geometry/coordinates/0':
            lon_col = col
        elif 'coordinates/1' in col or col == 'geometry/coordinates/1':
            lat_col = col
    
    if lon_col and lat_col:
        df['lon'] = df[lon_col]
        df['lat'] = df[lat_col]
    
    # Rename columns
    df.columns = df.columns.str.replace('properties/', '').str.replace('geometry/', '')
    df.columns = df.columns.str.replace('coordinates/', '')
    
    # Clean data
    if 'lon' in df.columns and 'lat' in df.columns:
        df = df.dropna(subset=['lon', 'lat'])
    
    # Fill missing severity
    if 'severity' in df.columns:
        df['severity'] = df['severity'].fillna(df['severity'].median())
    
    print(f"Loaded {len(df)} records")
    return df


def create_grid_cells_from_metadata(metadata: pd.DataFrame, 
                                     aggregation_metadata: Optional[pd.DataFrame] = None,
                                     risk_scores: Optional[np.ndarray] = None,
                                     hotspot_labels: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Create grid cells from aggregation metadata or compute from raw data"""
    
    if aggregation_metadata is not None and 'center_lon' in aggregation_metadata.columns:
        # Use existing aggregation metadata
        cells_df = aggregation_metadata.copy()
        
        if risk_scores is not None:
            cells_df['risk_score'] = risk_scores
        
        if hotspot_labels is not None:
            cells_df['is_hotspot'] = hotspot_labels
        
        # Compute risk level
        if 'risk_score' in cells_df.columns:
            risk_quantiles = cells_df['risk_score'].quantile([0.33, 0.67])
            def get_risk_level(score):
                if score >= risk_quantiles[0.67]:
                    return 'High'
                elif score >= risk_quantiles[0.33]:
                    return 'Medium'
                return 'Low'
            
            cells_df['risk_level'] = cells_df['risk_score'].apply(get_risk_level)
        else:
            cells_df['risk_level'] = 'Medium'
        
        return cells_df
    
    # Fallback: create grid from raw metadata
    print("Creating grid cells from raw metadata...")
    GRID_SIZE = 0.001  # ~100m
    
    min_lat, max_lat = metadata['lat'].min(), metadata['lat'].max()
    min_lon, max_lon = metadata['lon'].min(), metadata['lon'].max()
    
    metadata['grid_lat'] = ((metadata['lat'] - min_lat) / GRID_SIZE).astype(int)
    metadata['grid_lon'] = ((metadata['lon'] - min_lon) / GRID_SIZE).astype(int)
    metadata['cell_id'] = metadata['grid_lat'].astype(str) + '_' + metadata['grid_lon'].astype(str)
    
    cells = metadata.groupby('cell_id').agg({
        'lat': 'mean',
        'lon': 'mean',
        'severity': ['count', 'mean'],
        'label_type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown',
        'neighborhood': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
    }).reset_index()
    
    cells.columns = ['cell_id', 'lat', 'lon', 'barrier_count', 'severity_mean', 'dominant_type', 'neighborhood']
    
    # Risk level
    def get_risk(row):
        if row['barrier_count'] >= 50:
            return 'High'
        elif row['barrier_count'] >= 20:
            return 'Medium'
        return 'Low'
    
    cells['risk_level'] = cells.apply(get_risk, axis=1)
    cells['center_lat'] = cells['lat']
    cells['center_lon'] = cells['lon']
    
    return cells


def create_map_data(metadata: pd.DataFrame, 
                   aggregation_metadata: Optional[pd.DataFrame] = None,
                   risk_scores: Optional[np.ndarray] = None,
                   hotspot_labels: Optional[np.ndarray] = None,
                   gnn_results: Optional[Dict] = None) -> Dict:
    """Create JSON data for the interactive map"""
    print("Creating map data...")
    
    # Create grid cells
    cells_df = create_grid_cells_from_metadata(metadata, aggregation_metadata, risk_scores, hotspot_labels)
    
    # Sample barriers for map (performance)
    sample_size = min(5000, len(metadata))
    sample_barriers = metadata.sample(n=sample_size, random_state=42) if len(metadata) > sample_size else metadata
    
    barrier_points = []
    for _, row in sample_barriers.iterrows():
        barrier_points.append({
            'lat': round(row['lat'], 6),
            'lng': round(row['lon'], 6),
            'type': row.get('label_type', 'Unknown'),
            'severity': int(row.get('severity', 1)),
            'neighborhood': row.get('neighborhood', 'Unknown')
        })
    
    # Create cell data
    cell_data = []
    for idx, row in cells_df.iterrows():
        cell_id = row.get('grid_id', row.get('cell_id', f'cell_{idx}'))
        risk_score = row.get('risk_score', 0.5)
        is_hotspot = row.get('is_hotspot', row.get('hotspot', 0))
        
        cell_data.append({
            'lat': round(row.get('center_lat', row.get('lat', 0)), 6),
            'lng': round(row.get('center_lon', row.get('lon', 0)), 6),
            'cellId': str(cell_id),
            'barrierCount': int(row.get('count', row.get('barrier_count', 0))),
            'severityMean': round(row.get('severity_mean', row.get('severity', 1)), 2),
            'riskScore': round(float(risk_score), 4),
            'isHotspot': int(is_hotspot) if is_hotspot is not None else 0,
            'dominantType': row.get('dominant_type', 'Unknown'),
            'neighborhood': row.get('neighborhood', 'Unknown'),
            'riskLevel': row.get('risk_level', 'Medium')
        })
    
    # Neighborhood summary
    nb_summary = metadata.groupby('neighborhood').agg({
        'lat': 'mean',
        'lon': 'mean',
        'severity': ['count', 'mean']
    }).reset_index()
    nb_summary.columns = ['neighborhood', 'lat', 'lon', 'barrier_count', 'severity_mean']
    
    neighborhood_data = []
    for _, row in nb_summary.iterrows():
        neighborhood_data.append({
            'name': row['neighborhood'],
            'lat': round(row['lat'], 6),
            'lng': round(row['lon'], 6),
            'barrierCount': int(row['barrier_count']),
            'severityMean': round(row['severity_mean'], 2),
            'riskLevel': 'Medium'  # Default
        })
    
    # Statistics
    eval_data = gnn_results.get('evaluation', {}) if gnn_results else {}
    eval_metrics = eval_data.get('evaluation', {}) if isinstance(eval_data, dict) else {}
    
    stats = {
        'totalBarriers': len(metadata),
        'totalCells': len(cells_df),
        'totalNeighborhoods': len(nb_summary),
        'highRiskCells': len(cells_df[cells_df['risk_level'] == 'High']) if 'risk_level' in cells_df.columns else 0,
        'mediumRiskCells': len(cells_df[cells_df['risk_level'] == 'Medium']) if 'risk_level' in cells_df.columns else 0,
        'lowRiskCells': len(cells_df[cells_df['risk_level'] == 'Low']) if 'risk_level' in cells_df.columns else 0,
        'totalHotspots': int(eval_metrics.get('n_hotspots', 0)),
        'coverage': float(eval_metrics.get('coverage', {}).get('coverage', 0)) if isinstance(eval_metrics.get('coverage'), dict) else 0.0,
        'moransI': float(eval_metrics.get('morans_i', 0)) if eval_metrics.get('morans_i') is not None else None
    }
    
    map_data = {
        'barriers': barrier_points,
        'cells': cell_data,
        'neighborhoods': neighborhood_data,
        'stats': stats
    }
    
    return map_data


def main():
    """Main function to generate map data"""
    print("=" * 60)
    print(" GNN Map Data Generator")
    print("=" * 60)
    
    # Load raw data
    metadata = load_raw_data()
    
    if len(metadata) == 0:
        print("Warning: No data loaded. Creating empty map data structure.")
        map_data = {
            'barriers': [],
            'cells': [],
            'neighborhoods': [],
            'stats': {
                'totalBarriers': 0,
                'totalCells': 0,
                'totalNeighborhoods': 0,
                'highRiskCells': 0,
                'mediumRiskCells': 0,
                'lowRiskCells': 0,
                'totalHotspots': 0,
                'coverage': 0.0,
                'moransI': None
            }
        }
    else:
        # Load GNN results
        gnn_results = load_gnn_results()
        
        # Create map data
        map_data = create_map_data(metadata, gnn_results=gnn_results)
    
    # Save JSON
    output_path = OUTPUT_DIR / 'map_data.json'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(map_data, f, indent=2)
    
    print(f"\nSaved map data to: {output_path}")
    print(f"  Barriers: {len(map_data['barriers'])}")
    print(f"  Grid cells: {len(map_data['cells'])}")
    print(f"  Neighborhoods: {len(map_data['neighborhoods'])}")
    print(f"  Hotspots: {map_data['stats']['totalHotspots']}")


if __name__ == "__main__":
    main()

