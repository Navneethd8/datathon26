"""
Exploratory Data Analysis for Accessibility Dataset.

Answers key questions:
1. What types of barriers occur most frequently across Seattle?
2. Compare temporary vs. permanent barriers across geographic areas
3. Build an Accessibility Score for different neighborhoods

Also creates synthetic time bins for temporal modeling.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import (
    DATASET_PATH, PLOTS_DIR, NUM_TIME_BINS,
    LABEL_TYPE_COL, NEIGHBORHOOD_COL, SEVERITY_COL, 
    IS_TEMP_COL, ATTRIBUTE_ID_COL, COORD_COLS,
    LABEL_TYPES, BARRIER_WEIGHTS
)
from utils import setup_plotting, save_figure, print_section, print_subsection


def load_data() -> pd.DataFrame:
    """Load the accessibility dataset."""
    print(f"Loading data from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df):,} records with {df[NEIGHBORHOOD_COL].nunique()} neighborhoods")
    return df


def analyze_barrier_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Q1: What types of barriers occur most frequently across Seattle?
    """
    print_section("Q1: Barrier Type Frequency Analysis")
    
    # Overall frequency
    barrier_counts = df[LABEL_TYPE_COL].value_counts()
    barrier_pct = (barrier_counts / len(df) * 100).round(2)
    
    summary = pd.DataFrame({
        'Count': barrier_counts,
        'Percentage': barrier_pct
    })
    print(summary)
    print()
    
    # Visualize: Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall counts
    colors = sns.color_palette("viridis", len(barrier_counts))
    ax1 = axes[0]
    bars = ax1.barh(barrier_counts.index, barrier_counts.values, color=colors)
    ax1.set_xlabel('Count')
    ax1.set_title('Barrier Types: Overall Frequency')
    ax1.bar_label(bars, labels=[f'{v:,}' for v in barrier_counts.values], padding=3)
    
    # By neighborhood (top 10)
    ax2 = axes[1]
    top_neighborhoods = df[NEIGHBORHOOD_COL].value_counts().head(10).index
    df_top = df[df[NEIGHBORHOOD_COL].isin(top_neighborhoods)]
    
    pivot = df_top.pivot_table(
        index=NEIGHBORHOOD_COL, 
        columns=LABEL_TYPE_COL, 
        aggfunc='size', 
        fill_value=0
    )
    pivot = pivot.loc[top_neighborhoods]  # Keep order
    pivot.plot(kind='barh', stacked=True, ax=ax2, colormap='viridis')
    ax2.set_xlabel('Count')
    ax2.set_title('Barrier Types by Top 10 Neighborhoods')
    ax2.legend(title='Barrier Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    save_figure(fig, PLOTS_DIR, 'q1_barrier_type_frequency')
    
    return summary


def analyze_temp_vs_permanent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Q2: Compare temporary vs. permanent barriers across geographic areas.
    """
    print_section("Q2: Temporary vs Permanent Barriers")
    
    # Overall stats
    temp_counts = df[IS_TEMP_COL].value_counts()
    print(f"Permanent barriers: {temp_counts.get(False, 0):,} ({temp_counts.get(False, 0)/len(df)*100:.1f}%)")
    print(f"Temporary barriers: {temp_counts.get(True, 0):,} ({temp_counts.get(True, 0)/len(df)*100:.1f}%)")
    print()
    
    # By neighborhood
    neighborhood_temp = df.groupby(NEIGHBORHOOD_COL).agg({
        IS_TEMP_COL: ['sum', 'count']
    }).droplevel(0, axis=1)
    neighborhood_temp.columns = ['Temporary', 'Total']
    neighborhood_temp['Permanent'] = neighborhood_temp['Total'] - neighborhood_temp['Temporary']
    neighborhood_temp['Temp_Ratio'] = (neighborhood_temp['Temporary'] / neighborhood_temp['Total'] * 100).round(2)
    neighborhood_temp = neighborhood_temp.sort_values('Temp_Ratio', ascending=False)
    
    print_subsection("Top 10 Neighborhoods by Temporary Barrier Ratio")
    print(neighborhood_temp.head(10))
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Stacked bar: Top 15 neighborhoods
    ax1 = axes[0]
    top15 = neighborhood_temp.head(15)
    x = range(len(top15))
    ax1.barh(top15.index, top15['Permanent'], label='Permanent', color='#3498db')
    ax1.barh(top15.index, top15['Temporary'], left=top15['Permanent'], label='Temporary', color='#e74c3c')
    ax1.set_xlabel('Count')
    ax1.set_title('Temporary vs Permanent Barriers\n(Top 15 by Temp Ratio)')
    ax1.legend()
    ax1.invert_yaxis()
    
    # Temp ratio distribution
    ax2 = axes[1]
    ax2.hist(neighborhood_temp['Temp_Ratio'], bins=20, color='#9b59b6', edgecolor='white')
    ax2.axvline(neighborhood_temp['Temp_Ratio'].mean(), color='red', linestyle='--', 
                label=f"Mean: {neighborhood_temp['Temp_Ratio'].mean():.1f}%")
    ax2.set_xlabel('Temporary Barrier Ratio (%)')
    ax2.set_ylabel('Number of Neighborhoods')
    ax2.set_title('Distribution of Temporary Barrier Ratios')
    ax2.legend()
    
    save_figure(fig, PLOTS_DIR, 'q2_temp_vs_permanent')
    
    # Geographic heatmap (if we have enough variation)
    if df[IS_TEMP_COL].sum() > 100:
        fig, ax = plt.subplots(figsize=(12, 8))
        temp_only = df[df[IS_TEMP_COL] == True]
        scatter = ax.scatter(
            temp_only[COORD_COLS[0]], 
            temp_only[COORD_COLS[1]], 
            alpha=0.3, s=2, c='red'
        )
        perm_only = df[df[IS_TEMP_COL] == False].sample(min(5000, len(df)))
        ax.scatter(
            perm_only[COORD_COLS[0]], 
            perm_only[COORD_COLS[1]], 
            alpha=0.1, s=1, c='blue'
        )
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Geographic Distribution: Temporary (red) vs Permanent (blue) Barriers')
        save_figure(fig, PLOTS_DIR, 'q2_geographic_distribution')
    
    return neighborhood_temp


def compute_accessibility_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Q3: Build an Accessibility Score for different neighborhoods.
    
    Score formula (per neighborhood):
    - Higher score = WORSE accessibility (more barriers)
    - Score = sum(barrier_weight * severity) normalized by area/count
    """
    print_section("Q3: Accessibility Score by Neighborhood")
    
    # Fill missing severity with median
    df = df.copy()
    df[SEVERITY_COL] = df[SEVERITY_COL].fillna(df[SEVERITY_COL].median())
    
    # Calculate weighted barrier impact
    df['barrier_weight'] = df[LABEL_TYPE_COL].map(BARRIER_WEIGHTS).fillna(1.0)
    df['barrier_impact'] = df['barrier_weight'] * df[SEVERITY_COL]
    
    # Aggregate by neighborhood
    neighborhood_scores = df.groupby(NEIGHBORHOOD_COL).agg({
        'barrier_impact': 'sum',
        SEVERITY_COL: ['mean', 'max'],
        LABEL_TYPE_COL: 'count',
        IS_TEMP_COL: 'mean'
    })
    neighborhood_scores.columns = ['TotalImpact', 'MeanSeverity', 'MaxSeverity', 'BarrierCount', 'TempRatio']
    
    # Normalize accessibility score (0-100 scale, higher = worse)
    neighborhood_scores['AccessibilityScore'] = (
        (neighborhood_scores['TotalImpact'] / neighborhood_scores['BarrierCount']) * 10
    ).clip(0, 100).round(2)
    
    # Rank (1 = worst accessibility)
    neighborhood_scores['Rank'] = neighborhood_scores['AccessibilityScore'].rank(ascending=False).astype(int)
    neighborhood_scores = neighborhood_scores.sort_values('AccessibilityScore', ascending=False)
    
    print_subsection("Top 10 Neighborhoods with WORST Accessibility")
    print(neighborhood_scores[['AccessibilityScore', 'Rank', 'BarrierCount', 'MeanSeverity']].head(10))
    print()
    
    print_subsection("Top 10 Neighborhoods with BEST Accessibility")
    print(neighborhood_scores[['AccessibilityScore', 'Rank', 'BarrierCount', 'MeanSeverity']].tail(10))
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Score distribution
    ax1 = axes[0]
    ax1.hist(neighborhood_scores['AccessibilityScore'], bins=20, color='#e74c3c', edgecolor='white')
    ax1.axvline(neighborhood_scores['AccessibilityScore'].mean(), color='black', linestyle='--',
                label=f"Mean: {neighborhood_scores['AccessibilityScore'].mean():.1f}")
    ax1.set_xlabel('Accessibility Score (Higher = Worse)')
    ax1.set_ylabel('Number of Neighborhoods')
    ax1.set_title('Distribution of Accessibility Scores')
    ax1.legend()
    
    # Top/Bottom 10 comparison
    ax2 = axes[1]
    top10 = neighborhood_scores.head(10)
    bottom10 = neighborhood_scores.tail(10)
    
    y_labels = list(top10.index) + ['---'] + list(bottom10.index[::-1])
    scores = list(top10['AccessibilityScore']) + [0] + list(bottom10['AccessibilityScore'][::-1])
    colors = ['#e74c3c'] * 10 + ['white'] + ['#27ae60'] * 10
    
    ax2.barh(range(len(y_labels)), scores, color=colors)
    ax2.set_yticks(range(len(y_labels)))
    ax2.set_yticklabels(y_labels, fontsize=8)
    ax2.set_xlabel('Accessibility Score')
    ax2.set_title('Worst 10 (red) vs Best 10 (green) Neighborhoods')
    ax2.invert_yaxis()
    
    save_figure(fig, PLOTS_DIR, 'q3_accessibility_scores')
    
    # Save scores to CSV
    output_path = PLOTS_DIR.parent / 'accessibility_scores.csv'
    neighborhood_scores.to_csv(output_path)
    print(f"\nSaved accessibility scores to {output_path}")
    
    return neighborhood_scores


def create_synthetic_time_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic time bins using attribute_id ordering.
    Assumes attribute_id was assigned chronologically.
    """
    print_section("Creating Synthetic Time Bins")
    
    df = df.copy()
    df = df.sort_values(ATTRIBUTE_ID_COL).reset_index(drop=True)
    
    # Create time bins
    df['time_bin'] = pd.qcut(df.index, q=NUM_TIME_BINS, labels=False)
    
    print(f"Created {NUM_TIME_BINS} time bins")
    print(f"Records per bin: ~{len(df) // NUM_TIME_BINS:,}")
    
    # Analyze trends over time bins
    time_trends = df.groupby('time_bin').agg({
        LABEL_TYPE_COL: 'count',
        SEVERITY_COL: 'mean',
        IS_TEMP_COL: 'mean'
    })
    time_trends.columns = ['BarrierCount', 'MeanSeverity', 'TempRatio']
    
    # Plot time trends
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    ax1 = axes[0]
    ax1.plot(time_trends.index, time_trends['BarrierCount'], marker='o', color='#3498db')
    ax1.set_ylabel('Barrier Count')
    ax1.set_title('Synthetic Time Series: Barrier Metrics Over Time Bins')
    ax1.fill_between(time_trends.index, time_trends['BarrierCount'], alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(time_trends.index, time_trends['MeanSeverity'], marker='s', color='#e74c3c')
    ax2.set_ylabel('Mean Severity')
    ax2.fill_between(time_trends.index, time_trends['MeanSeverity'], alpha=0.3, color='#e74c3c')
    
    ax3 = axes[2]
    ax3.plot(time_trends.index, time_trends['TempRatio'] * 100, marker='^', color='#9b59b6')
    ax3.set_ylabel('Temp Ratio (%)')
    ax3.set_xlabel('Time Bin')
    ax3.fill_between(time_trends.index, time_trends['TempRatio'] * 100, alpha=0.3, color='#9b59b6')
    
    save_figure(fig, PLOTS_DIR, 'synthetic_time_series')
    
    # Barrier type trends
    type_trends = df.pivot_table(
        index='time_bin',
        columns=LABEL_TYPE_COL,
        aggfunc='size',
        fill_value=0
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    type_trends.plot(ax=ax, colormap='viridis', marker='o', markersize=3)
    ax.set_xlabel('Time Bin')
    ax.set_ylabel('Count')
    ax.set_title('Barrier Types Over Synthetic Time Bins')
    ax.legend(title='Barrier Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    save_figure(fig, PLOTS_DIR, 'barrier_types_over_time')
    
    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print overall summary statistics."""
    print_section("Summary Statistics")
    
    print(f"Total Records: {len(df):,}")
    print(f"Total Neighborhoods: {df[NEIGHBORHOOD_COL].nunique()}")
    print(f"Label Types: {df[LABEL_TYPE_COL].nunique()}")
    print()
    
    print_subsection("Severity Distribution")
    print(df[SEVERITY_COL].describe())
    print()
    
    print_subsection("Records per Neighborhood")
    print(df[NEIGHBORHOOD_COL].value_counts().describe())


def main():
    """Run the complete EDA pipeline."""
    setup_plotting()
    
    # Load data
    df = load_data()
    
    # Summary stats
    print_summary_statistics(df)
    
    # Q1: Barrier types
    barrier_summary = analyze_barrier_types(df)
    
    # Q2: Temp vs permanent
    temp_analysis = analyze_temp_vs_permanent(df)
    
    # Q3: Accessibility scores
    accessibility_scores = compute_accessibility_scores(df)
    
    # Create synthetic time bins for temporal modeling
    df_with_bins = create_synthetic_time_bins(df)
    
    print_section("EDA Complete")
    print(f"All plots saved to: {PLOTS_DIR}")
    print(f"Accessibility scores saved to: {PLOTS_DIR.parent / 'accessibility_scores.csv'}")
    
    return df_with_bins, accessibility_scores


if __name__ == "__main__":
    main()
