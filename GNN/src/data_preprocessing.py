"""
Data Preprocessing Module
Loads and preprocesses accessibility data for GNN training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocesses accessibility data for graph neural network"""
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scaler_lon = StandardScaler()
        self.scaler_lat = StandardScaler()
        self.scaler_severity = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.neighborhood_encoder = LabelEncoder()
        
    def load_data(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load CSV data
        
        Args:
            csv_path: Path to CSV file (uses config if None)
            
        Returns:
            DataFrame with loaded data
        """
        if csv_path is None:
            csv_path = self.config['data']['csv_path']
        
        # Resolve relative path
        if csv_path.startswith('../'):
            base_path = Path(__file__).parent.parent.parent
            csv_path = base_path / csv_path
        
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        # Extract coordinate columns BEFORE renaming
        # The CSV has columns like: geometry/coordinates/0 and geometry/coordinates/1
        lon_col = None
        lat_col = None
        
        for col in df.columns:
            if 'coordinates/0' in col or col == 'geometry/coordinates/0':
                lon_col = col
            elif 'coordinates/1' in col or col == 'geometry/coordinates/1':
                lat_col = col
        
        # If exact match not found, try pattern matching
        if lon_col is None:
            coord_cols = [c for c in df.columns if 'coordinate' in c.lower() and ('0' in c or '/0' in c)]
            if coord_cols:
                lon_col = coord_cols[0]
        
        if lat_col is None:
            coord_cols = [c for c in df.columns if 'coordinate' in c.lower() and ('1' in c or '/1' in c)]
            if coord_cols:
                lat_col = coord_cols[0]
        
        # Extract coordinates
        if lon_col is not None:
            df['lon'] = pd.to_numeric(df[lon_col], errors='coerce')
        else:
            raise ValueError(f"Could not find longitude column. Available columns: {list(df.columns)}")
        
        if lat_col is not None:
            df['lat'] = pd.to_numeric(df[lat_col], errors='coerce')
        else:
            raise ValueError(f"Could not find latitude column. Available columns: {list(df.columns)}")
        
        # Now rename columns for easier access (after extracting coordinates)
        df.columns = df.columns.str.replace('properties/', '').str.replace('geometry/', '')
        df.columns = df.columns.str.replace('coordinates/', '')
        
        # Filter by city if specified
        if not self.config['data']['process_all_cities']:
            city_filter = self.config['data'].get('city_filter')
            if city_filter:
                # This would require additional data - for now process all
                pass
        
        # Handle missing severity
        df['severity'] = pd.to_numeric(df['severity'], errors='coerce')
        missing_severity = df['severity'].isna().sum()
        if missing_severity > 0:
            print(f"Found {missing_severity} records with missing severity")
            # Impute with median
            median_severity = df['severity'].median()
            df['severity'].fillna(median_severity, inplace=True)
            print(f"Imputed missing severity with median: {median_severity}")
        
        # Handle missing neighborhood
        df['neighborhood'] = df['neighborhood'].fillna('Unknown')
        
        # Convert is_temporary to boolean
        if 'is_temporary' in df.columns:
            df['is_temporary'] = df['is_temporary'].astype(bool)
        else:
            df['is_temporary'] = False
        
        # Remove any rows with missing coordinates
        df = df.dropna(subset=['lon', 'lat'])
        
        print(f"After cleaning: {len(df)} records")
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Create feature vectors for each point
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            Tuple of (feature_matrix, metadata_dataframe)
        """
        df = df.copy()
        
        # One-hot encode label_type
        label_types = ['SurfaceProblem', 'NoCurbRamp', 'CurbRamp']
        for label_type in label_types:
            df[f'label_{label_type}'] = (df['label_type'] == label_type).astype(float)
        
        # Normalize severity (1-5 scale)
        df['severity_normalized'] = (df['severity'] - 1) / 4.0  # Scale to [0, 1]
        
        # Encode neighborhood (optional - can use one-hot or embedding)
        # For now, we'll use a simple numeric encoding
        if 'neighborhood' in df.columns:
            df['neighborhood_encoded'] = self.neighborhood_encoder.fit_transform(df['neighborhood'])
            # Normalize neighborhood encoding
            if df['neighborhood_encoded'].max() > 0:
                df['neighborhood_encoded'] = df['neighborhood_encoded'] / df['neighborhood_encoded'].max()
        
        # Normalize coordinates
        df['lon_normalized'] = self.scaler_lon.fit_transform(df[['lon']])
        df['lat_normalized'] = self.scaler_lat.fit_transform(df[['lat']])
        
        # Convert is_temporary to float
        df['is_temporary_float'] = df['is_temporary'].astype(float)
        
        # Create feature matrix
        feature_cols = [
            'label_SurfaceProblem', 'label_NoCurbRamp', 'label_CurbRamp',
            'severity_normalized',
            'is_temporary_float',
            'lon_normalized', 'lat_normalized'
        ]
        
        # Add neighborhood if available
        if 'neighborhood_encoded' in df.columns:
            feature_cols.append('neighborhood_encoded')
        
        feature_matrix = df[feature_cols].values.astype(np.float32)
        
        # Create metadata dataframe
        metadata = df[['lon', 'lat', 'label_type', 'severity', 'neighborhood', 
                       'is_temporary', 'attribute_id']].copy()
        
        print(f"Created feature matrix with shape: {feature_matrix.shape}")
        print(f"Feature columns: {feature_cols}")
        
        return feature_matrix, metadata
    
    def process(self, csv_path: Optional[str] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Complete preprocessing pipeline
        
        Args:
            csv_path: Optional path to CSV file
            
        Returns:
            Tuple of (feature_matrix, metadata_dataframe)
        """
        df = self.load_data(csv_path)
        df = self.clean_data(df)
        features, metadata = self.create_features(df)
        
        return features, metadata
    
    def get_feature_dim(self) -> int:
        """
        Get the dimension of feature vectors
        
        Returns:
            Feature dimension
        """
        # 3 (label types) + 1 (severity) + 1 (is_temporary) + 2 (coords) + 1 (neighborhood if used)
        base_dim = 7
        if hasattr(self, 'neighborhood_encoder') and len(self.neighborhood_encoder.classes_) > 0:
            return base_dim + 1
        return base_dim

