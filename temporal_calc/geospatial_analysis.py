"""
Geospatial Analysis Module.

Adds:
1. Geospatial heatmaps using coordinates
2. Accessibility Density (barriers per km²)
3. Demographic data for equity analysis
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

from config import (
    DATA_DIR, PLOTS_DIR, OUTPUT_DIR, COORD_COLS, 
    NEIGHBORHOOD_COL, LABEL_TYPE_COL, SEVERITY_COL,
)
from utils import setup_plotting, save_figure, print_section, print_subsection


# Seattle neighborhood approximate demographics (2020 Census / ACS estimates)
# Source: Seattle Office of Planning & Community Development
SEATTLE_DEMOGRAPHICS = {
    "Atlantic": {"population": 8500, "median_income": 65000, "pct_minority": 45.2, "pct_elderly": 12.1},
    "Ballard": {"population": 52000, "median_income": 95000, "pct_minority": 18.5, "pct_elderly": 10.2},
    "Beacon Hill": {"population": 32000, "median_income": 55000, "pct_minority": 68.4, "pct_elderly": 14.8},
    "Belltown": {"population": 12000, "median_income": 82000, "pct_minority": 22.3, "pct_elderly": 5.4},
    "Briarcliff": {"population": 4200, "median_income": 88000, "pct_minority": 15.2, "pct_elderly": 11.5},
    "Capitol Hill": {"population": 38000, "median_income": 78000, "pct_minority": 21.8, "pct_elderly": 8.9},
    "Central Business District": {"population": 3500, "median_income": 95000, "pct_minority": 35.2, "pct_elderly": 4.2},
    "Columbia City": {"population": 15000, "median_income": 62000, "pct_minority": 58.3, "pct_elderly": 11.2},
    "First Hill": {"population": 18000, "median_income": 48000, "pct_minority": 42.5, "pct_elderly": 18.5},
    "Fremont": {"population": 14000, "median_income": 98000, "pct_minority": 16.8, "pct_elderly": 7.3},
    "Georgetown": {"population": 2800, "median_income": 52000, "pct_minority": 48.6, "pct_elderly": 9.8},
    "Green Lake": {"population": 18000, "median_income": 92000, "pct_minority": 14.2, "pct_elderly": 11.8},
    "Greenwood": {"population": 22000, "median_income": 85000, "pct_minority": 19.5, "pct_elderly": 12.4},
    "Industrial District": {"population": 1200, "median_income": 45000, "pct_minority": 52.3, "pct_elderly": 8.2},
    "Interbay": {"population": 3500, "median_income": 72000, "pct_minority": 24.1, "pct_elderly": 9.5},
    "Laurelhurst": {"population": 8500, "median_income": 185000, "pct_minority": 12.4, "pct_elderly": 15.8},
    "Leschi": {"population": 6200, "median_income": 125000, "pct_minority": 28.3, "pct_elderly": 13.2},
    "Loyal Heights": {"population": 9500, "median_income": 115000, "pct_minority": 15.8, "pct_elderly": 14.2},
    "Madrona": {"population": 7800, "median_income": 142000, "pct_minority": 32.5, "pct_elderly": 11.8},
    "Magnolia": {"population": 24000, "median_income": 125000, "pct_minority": 11.2, "pct_elderly": 16.5},
    "Mount Baker": {"population": 11000, "median_income": 78000, "pct_minority": 55.8, "pct_elderly": 12.4},
    "North Queen Anne": {"population": 15000, "median_income": 105000, "pct_minority": 18.5, "pct_elderly": 10.8},
    "Phinney Ridge": {"population": 12000, "median_income": 98000, "pct_minority": 14.8, "pct_elderly": 12.2},
    "Rainier Beach": {"population": 18000, "median_income": 48000, "pct_minority": 82.5, "pct_elderly": 10.5},
    "Ravenna": {"population": 14000, "median_income": 115000, "pct_minority": 18.2, "pct_elderly": 14.8},
    "Roosevelt": {"population": 8500, "median_income": 88000, "pct_minority": 22.5, "pct_elderly": 11.2},
    "South Lake Union": {"population": 8000, "median_income": 125000, "pct_minority": 28.5, "pct_elderly": 4.8},
    "Stevens": {"population": 5200, "median_income": 165000, "pct_minority": 15.2, "pct_elderly": 14.5},
    "Sunset Hill": {"population": 6800, "median_income": 135000, "pct_minority": 12.8, "pct_elderly": 15.2},
    "University District": {"population": 32000, "median_income": 42000, "pct_minority": 38.5, "pct_elderly": 6.2},
    "View Ridge": {"population": 7500, "median_income": 145000, "pct_minority": 14.5, "pct_elderly": 16.8},
    "Wallingford": {"population": 18000, "median_income": 105000, "pct_minority": 16.2, "pct_elderly": 10.5},
    "West Woodland": {"population": 8500, "median_income": 95000, "pct_minority": 15.8, "pct_elderly": 11.8},
    "Westlake": {"population": 4200, "median_income": 78000, "pct_minority": 32.5, "pct_elderly": 8.5},
}


def load_geojson_data() -> pd.DataFrame:
    """Load data from attributes-seattle.json GeoJSON file."""
    json_path = DATA_DIR / "attributes-seattle.json"
    
    print(f"Loading GeoJSON from {json_path}...")
    
    with open(json_path, 'r') as f:
        geojson = json.load(f)
    
    records = []
    for feature in geojson['features']:
        props = feature['properties']
        coords = feature['geometry']['coordinates']
        records.append({
            'longitude': coords[0],
            'latitude': coords[1],
            'attribute_id': props['attribute_id'],
            'label_type': props['label_type'],
            'neighborhood': props['neighborhood'],
            'severity': props.get('severity'),
            'is_temporary': props['is_temporary']
        })
    
    df = pd.DataFrame(records)
    print(f"Loaded {len(df):,} features from GeoJSON")
    return df


def calculate_neighborhood_areas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate approximate neighborhood areas using bounding box method.
    Returns area in km².
    """
    print_subsection("Calculating Neighborhood Areas")
    
    areas = []
    for neighborhood in df['neighborhood'].unique():
        nb_data = df[df['neighborhood'] == neighborhood]
        
        # Get bounding box
        min_lon, max_lon = nb_data['longitude'].min(), nb_data['longitude'].max()
        min_lat, max_lat = nb_data['latitude'].min(), nb_data['latitude'].max()
        
        # Convert to km (approximate at Seattle's latitude ~47.6°N)
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 * cos(47.6°) ≈ 75 km
        lat_km = (max_lat - min_lat) * 111
        lon_km = (max_lon - min_lon) * 75
        
        # Bounding box area (multiply by ~0.7 to estimate actual coverage)
        area_km2 = lat_km * lon_km * 0.7
        
        # Minimum area to avoid division issues
        area_km2 = max(area_km2, 0.1)
        
        areas.append({
            'neighborhood': neighborhood,
            'area_km2': area_km2,
            'min_lon': min_lon,
            'max_lon': max_lon,
            'min_lat': min_lat,
            'max_lat': max_lat,
            'center_lon': (min_lon + max_lon) / 2,
            'center_lat': (min_lat + max_lat) / 2
        })
    
    return pd.DataFrame(areas)


def compute_accessibility_density(df: pd.DataFrame, areas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accessibility density = barriers per km² for each neighborhood.
    """
    print_subsection("Computing Accessibility Density")
    
    # Count barriers per neighborhood
    barrier_counts = df.groupby('neighborhood').agg({
        'attribute_id': 'count',
        'severity': 'mean'
    }).reset_index()
    barrier_counts.columns = ['neighborhood', 'barrier_count', 'avg_severity']
    
    # Merge with areas
    density_df = barrier_counts.merge(areas_df, on='neighborhood')
    
    # Calculate density
    density_df['barriers_per_km2'] = density_df['barrier_count'] / density_df['area_km2']
    
    # Calculate weighted density (by severity)
    density_df['severity_weighted_density'] = (
        density_df['barrier_count'] * density_df['avg_severity'].fillna(3)
    ) / density_df['area_km2']
    
    return density_df.sort_values('barriers_per_km2', ascending=False)


def add_demographic_data(density_df: pd.DataFrame) -> pd.DataFrame:
    """Add demographic data for equity analysis."""
    print_subsection("Adding Demographic Data")
    
    # Create demographics DataFrame
    demo_records = []
    for nb, data in SEATTLE_DEMOGRAPHICS.items():
        demo_records.append({
            'neighborhood': nb,
            'population': data['population'],
            'median_income': data['median_income'],
            'pct_minority': data['pct_minority'],
            'pct_elderly': data['pct_elderly']
        })
    
    demo_df = pd.DataFrame(demo_records)
    
    # Merge with density data
    equity_df = density_df.merge(demo_df, on='neighborhood', how='left')
    
    # Calculate per-capita metrics
    equity_df['barriers_per_1000_pop'] = (
        equity_df['barrier_count'] / equity_df['population'] * 1000
    ).fillna(0)
    
    # Equity score: higher = worse equity (more barriers in vulnerable areas)
    # Weighted by minority %, elderly %, inverse of income
    income_factor = equity_df['median_income'].max() / equity_df['median_income'].fillna(equity_df['median_income'].median())
    
    equity_df['equity_concern_score'] = (
        equity_df['barriers_per_km2'] * 
        (1 + equity_df['pct_minority'].fillna(0) / 100) *
        (1 + equity_df['pct_elderly'].fillna(0) / 100) *
        income_factor.fillna(1)
    )
    
    return equity_df


def generate_geospatial_heatmap(df: pd.DataFrame, density_df: pd.DataFrame):
    """Generate geospatial heatmap visualizations."""
    print_subsection("Generating Geospatial Heatmaps")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Barrier Location Scatter (Heatmap-style)
    ax1 = axes[0, 0]
    severity = df['severity'].fillna(3)
    scatter = ax1.scatter(
        df['longitude'], df['latitude'],
        c=severity, cmap='YlOrRd', alpha=0.4, s=5
    )
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Barrier Locations by Severity\n(Red = High Severity)')
    plt.colorbar(scatter, ax=ax1, label='Severity')
    
    # 2. Neighborhood Density Map
    ax2 = axes[0, 1]
    for _, row in density_df.iterrows():
        color_val = row['barriers_per_km2'] / density_df['barriers_per_km2'].max()
        ax2.scatter(
            row['center_lon'], row['center_lat'],
            s=row['barrier_count'] / 5,
            c=[plt.cm.Reds(color_val)],
            alpha=0.7, edgecolors='black', linewidth=0.5
        )
        # Label top 10
        if row['barriers_per_km2'] >= density_df['barriers_per_km2'].nlargest(10).min():
            ax2.annotate(
                row['neighborhood'][:12],
                (row['center_lon'], row['center_lat']),
                fontsize=7, alpha=0.8
            )
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Accessibility Density by Neighborhood\n(Size = Count, Color = Density)')
    
    # 3. Barrier Type Distribution by Location
    ax3 = axes[1, 0]
    barrier_types = df['label_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(barrier_types)))
    for bt, color in zip(barrier_types, colors):
        bt_data = df[df['label_type'] == bt]
        ax3.scatter(
            bt_data['longitude'], bt_data['latitude'],
            c=[color], alpha=0.3, s=3, label=bt
        )
    ax3.legend(loc='upper left', fontsize=7, markerscale=3)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title('Barrier Types by Location')
    
    # 4. High-Density Hotspots
    ax4 = axes[1, 1]
    top_density = density_df.nlargest(15, 'barriers_per_km2')
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(top_density)))
    bars = ax4.barh(
        top_density['neighborhood'],
        top_density['barriers_per_km2'],
        color=colors
    )
    ax4.set_xlabel('Barriers per km²')
    ax4.set_title('Top 15 Neighborhoods by Accessibility Density')
    ax4.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, top_density['barriers_per_km2']):
        ax4.text(val + 5, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', fontsize=8)
    
    plt.suptitle('Seattle Accessibility Geospatial Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, PLOTS_DIR, 'geospatial_heatmap')
    
    return fig


def generate_equity_analysis(equity_df: pd.DataFrame):
    """Generate equity analysis visualizations."""
    print_subsection("Generating Equity Analysis")
    
    # Filter to neighborhoods with demographic data
    eq = equity_df.dropna(subset=['population'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Barriers vs Minority Population
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        eq['pct_minority'], eq['barriers_per_km2'],
        c=eq['median_income'], cmap='viridis_r', s=eq['population']/200,
        alpha=0.7, edgecolors='black', linewidth=0.5
    )
    ax1.set_xlabel('% Minority Population')
    ax1.set_ylabel('Barriers per km²')
    ax1.set_title('Accessibility vs Demographic Composition\n(Size = Population, Color = Income)')
    plt.colorbar(scatter, ax=ax1, label='Median Income ($)')
    
    # Add trend line
    z = np.polyfit(eq['pct_minority'], eq['barriers_per_km2'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(eq['pct_minority'].min(), eq['pct_minority'].max(), 100)
    ax1.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Trend')
    ax1.legend()
    
    # 2. Income vs Barrier Density
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(
        eq['median_income'] / 1000, eq['barriers_per_km2'],
        c=eq['pct_minority'], cmap='coolwarm', s=100, alpha=0.7
    )
    ax2.set_xlabel('Median Income ($K)')
    ax2.set_ylabel('Barriers per km²')
    ax2.set_title('Income vs Accessibility Barriers\n(Color = % Minority)')
    plt.colorbar(scatter2, ax=ax2, label='% Minority')
    
    # 3. Equity Concern Score Rankings
    ax3 = axes[1, 0]
    top_concern = eq.nlargest(15, 'equity_concern_score')
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(top_concern)))
    bars = ax3.barh(
        top_concern['neighborhood'],
        top_concern['equity_concern_score'],
        color=colors
    )
    ax3.set_xlabel('Equity Concern Score')
    ax3.set_title('Top 15 Neighborhoods: Equity Concern\n(High = More barriers + Vulnerable population)')
    ax3.invert_yaxis()
    
    # 4. Barriers per 1000 Population
    ax4 = axes[1, 1]
    top_percap = eq.nlargest(15, 'barriers_per_1000_pop')
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(top_percap)))
    bars = ax4.barh(
        top_percap['neighborhood'],
        top_percap['barriers_per_1000_pop'],
        color=colors
    )
    ax4.set_xlabel('Barriers per 1,000 Population')
    ax4.set_title('Top 15 Neighborhoods: Per-Capita Barrier Burden')
    ax4.invert_yaxis()
    
    plt.suptitle('Equity Analysis: Accessibility & Demographics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, PLOTS_DIR, 'equity_analysis')
    
    return fig


def main():
    """Run full geospatial analysis pipeline."""
    setup_plotting()
    print_section("Geospatial Analysis Pipeline")
    
    # 1. Load GeoJSON data
    df = load_geojson_data()
    
    # 2. Calculate neighborhood areas
    areas_df = calculate_neighborhood_areas(df)
    
    # 3. Compute accessibility density
    density_df = compute_accessibility_density(df, areas_df)
    
    print("\n--- Top 10 Highest Density Neighborhoods ---")
    print(density_df[['neighborhood', 'barrier_count', 'area_km2', 'barriers_per_km2']].head(10).to_string(index=False))
    
    # 4. Add demographic data for equity analysis
    equity_df = add_demographic_data(density_df)
    
    # 5. Generate visualizations
    generate_geospatial_heatmap(df, density_df)
    generate_equity_analysis(equity_df)
    
    # 6. Save outputs
    output_path = OUTPUT_DIR / 'geospatial_equity_analysis.csv'
    equity_df.to_csv(output_path, index=False)
    print(f"\nSaved analysis to: {output_path}")
    
    # 7. Print equity summary
    print_subsection("Equity Summary")
    eq = equity_df.dropna(subset=['population'])
    if len(eq) > 0:
        correlation = eq['pct_minority'].corr(eq['barriers_per_km2'])
        income_corr = eq['median_income'].corr(eq['barriers_per_km2'])
        print(f"Correlation (% Minority vs Barrier Density): {correlation:.3f}")
        print(f"Correlation (Income vs Barrier Density): {income_corr:.3f}")
        print(f"\nTop 5 Equity Concern Neighborhoods:")
        print(eq.nlargest(5, 'equity_concern_score')[
            ['neighborhood', 'pct_minority', 'median_income', 'equity_concern_score']
        ].to_string(index=False))
    
    print_section("Geospatial Analysis Complete")
    
    return equity_df


if __name__ == "__main__":
    main()
