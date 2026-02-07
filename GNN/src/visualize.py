"""
Visualization Module
Creates interactive maps and visualizations
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import folium
    from folium.plugins import HeatMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    print("Warning: folium not available. Using matplotlib fallback.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not available. Using matplotlib fallback.")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class Visualizer:
    """Create visualizations for hotspot detection"""
    
    def __init__(self, config: Dict, output_dir: str = "outputs"):
        """
        Initialize visualizer
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for visualizations
        """
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_interactive_map_folium(self, metadata: pd.DataFrame,
                                     hotspot_results: Dict,
                                     aggregation_metadata: Optional[pd.DataFrame] = None) -> folium.Map:
        """
        Create interactive map using Folium
        
        Args:
            metadata: Original point metadata
            hotspot_results: Hotspot detection results
            aggregation_metadata: Aggregated spatial unit metadata
            
        Returns:
            Folium map object
        """
        if not HAS_FOLIUM:
            raise ImportError("folium is required for interactive maps")
        
        # Center map on data
        center_lat = metadata['lat'].mean()
        center_lon = metadata['lon'].mean()
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles=self.viz_config.get('map_style', 'OpenStreetMap')
        )
        
        # Add all points colored by type/severity
        point_size = self.viz_config.get('point_size', 3)
        
        # Color map for label types
        color_map = {
            'SurfaceProblem': 'red',
            'NoCurbRamp': 'orange',
            'CurbRamp': 'green'
        }
        
        # Sample points if too many
        max_points = 5000
        if len(metadata) > max_points:
            metadata_sample = metadata.sample(n=max_points, random_state=42)
        else:
            metadata_sample = metadata
        
        for _, row in metadata_sample.iterrows():
            color = color_map.get(row['label_type'], 'gray')
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=point_size,
                popup=f"Type: {row['label_type']}, Severity: {row['severity']}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(m)
        
        # Add hotspots if available
        if aggregation_metadata is not None and 'hotspot' in aggregation_metadata.columns:
            hotspot_units = aggregation_metadata[aggregation_metadata['hotspot'] == 1]
            
            # Create heatmap data
            heat_data = [[row['center_lat'], row['center_lon'], row.get('risk_score', 1.0)]
                        for _, row in hotspot_units.iterrows()]
            
            if heat_data:
                HeatMap(
                    heat_data,
                    min_opacity=self.viz_config.get('heatmap_opacity', 0.6),
                    radius=15,
                    blur=10,
                    max_zoom=1
                ).add_to(m)
        
        return m
    
    def create_interactive_map_plotly(self, metadata: pd.DataFrame,
                                      hotspot_results: Dict,
                                      aggregation_metadata: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create interactive map using Plotly
        
        Args:
            metadata: Original point metadata
            hotspot_results: Hotspot detection results
            aggregation_metadata: Aggregated spatial unit metadata
            
        Returns:
            Plotly figure object
        """
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for interactive maps")
        
        # Sample if too many points
        max_points = 5000
        if len(metadata) > max_points:
            metadata_sample = metadata.sample(n=max_points, random_state=42)
        else:
            metadata_sample = metadata
        
        # Create scatter plot of points
        fig = px.scatter_mapbox(
            metadata_sample,
            lat='lat',
            lon='lon',
            color='label_type',
            size='severity',
            hover_data=['severity', 'neighborhood'],
            zoom=12,
            height=600
        )
        
        # Add hotspots if available
        if aggregation_metadata is not None and 'hotspot' in aggregation_metadata.columns:
            hotspot_units = aggregation_metadata[aggregation_metadata['hotspot'] == 1]
            
            fig.add_trace(go.Scattermapbox(
                lat=hotspot_units['center_lat'],
                lon=hotspot_units['center_lon'],
                mode='markers',
                marker=dict(
                    size=20,
                    color='red',
                    opacity=0.7
                ),
                name='Hotspots',
                text=hotspot_units.get('risk_score', '')
            ))
        
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        
        return fig
    
    def plot_embeddings(self, embeddings: np.ndarray, labels: np.ndarray,
                       save_path: Optional[str] = None):
        """
        Plot embeddings using t-SNE or UMAP
        
        Args:
            embeddings: Node embeddings
            labels: Cluster or hotspot labels
            save_path: Path to save plot
        """
        try:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = reducer.fit_transform(embeddings)
        except Exception:
            # Fallback: use first two dimensions
            embeddings_2d = embeddings[:, :2]
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Node Embeddings (t-SNE)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Embeddings plot saved to {save_path}")
        else:
            plt.savefig(self.output_dir / 'embeddings.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize(self, metadata: pd.DataFrame, hotspot_results: Dict,
                  embeddings: Optional[np.ndarray] = None,
                  aggregation_metadata: Optional[pd.DataFrame] = None):
        """
        Create all visualizations
        
        Args:
            metadata: Original point metadata
            hotspot_results: Hotspot detection results
            embeddings: Optional node embeddings
            aggregation_metadata: Aggregated spatial unit metadata
        """
        print("Creating visualizations...")
        
        # Interactive map
        try:
            if HAS_FOLIUM:
                m = self.create_interactive_map_folium(metadata, hotspot_results, aggregation_metadata)
                map_path = self.output_dir / 'hotspot_map.html'
                m.save(str(map_path))
                print(f"Interactive map saved to {map_path}")
            elif HAS_PLOTLY:
                fig = self.create_interactive_map_plotly(metadata, hotspot_results, aggregation_metadata)
                map_path = self.output_dir / 'hotspot_map.html'
                fig.write_html(str(map_path))
                print(f"Interactive map saved to {map_path}")
        except Exception as e:
            print(f"Error creating interactive map: {e}")
        
        # Embeddings plot
        if embeddings is not None and self.viz_config.get('save_embeddings_plot', True):
            cluster_labels = hotspot_results.get('cluster_labels', 
                                                 np.zeros(len(embeddings)))
            self.plot_embeddings(embeddings, cluster_labels)
        
        print("Visualization complete!")

