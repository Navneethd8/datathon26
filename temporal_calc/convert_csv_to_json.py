"""
Helper script to convert CSV files in outputs/ to JSON format for easier loading in the dashboard.

This is optional - the dashboard can load CSV files directly, but JSON is more reliable.
"""
import json
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"

def convert_csv_to_json(csv_path, json_path=None):
    """Convert a CSV file to JSON format."""
    if json_path is None:
        json_path = csv_path.with_suffix('.json')
    
    try:
        df = pd.read_csv(csv_path)
        # Convert to list of dictionaries
        data = df.to_dict('records')
        
        # Save as JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Converted {csv_path.name} -> {json_path.name} ({len(data)} records)")
        return True
    except Exception as e:
        print(f"✗ Error converting {csv_path.name}: {e}")
        return False

def main():
    """Convert all CSV files in outputs/ to JSON."""
    print("Converting CSV files to JSON for dashboard...")
    print()
    
    csv_files = [
        OUTPUT_DIR / "neighborhood_risk_forecast.csv",
        OUTPUT_DIR / "accessibility_scores.csv",
        OUTPUT_DIR / "eval" / "metrics.csv",
    ]
    
    converted = 0
    for csv_file in csv_files:
        if csv_file.exists():
            if convert_csv_to_json(csv_file):
                converted += 1
        else:
            print(f"Warning: {csv_file.name} not found, skipping")
    
    print()
    print(f"Converted {converted}/{len(csv_files)} files")
    print("\nNote: The dashboard can load CSV files directly, but JSON is more reliable.")

if __name__ == "__main__":
    main()

