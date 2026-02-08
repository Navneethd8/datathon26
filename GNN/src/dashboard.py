"""
Comprehensive Dashboard for GNN Model and Evaluation Metrics
Creates interactive and static visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not available. Some interactive dashboards will be skipped.")

try:
    import dash
    from dash import dcc, html, Input, Output
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    print("Warning: dash not available. Interactive web dashboard will be skipped.")


class ModelDashboard:
    """Create comprehensive dashboards for model and evaluation metrics"""
    
    def __init__(self, output_dir: str = "outputs/dashboards"):
        """
        Initialize dashboard creator
        
        Args:
            output_dir: Directory to save dashboard files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def load_results(self, results_file: str = "outputs/results.json") -> Optional[Dict]:
        """Load results from JSON file"""
        results_path = Path(results_file)
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        return None
    
    def create_training_curves(self, history: Dict, save_path: Optional[str] = None):
        """
        Create training loss curves
        
        Args:
            history: Training history dictionary with 'train_loss' key
            save_path: Path to save the plot
        """
        if 'train_loss' not in history or len(history['train_loss']) == 0:
            print("No training history available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
        
        # Add moving average
        if len(history['train_loss']) > 10:
            window = min(10, len(history['train_loss']) // 10)
            moving_avg = pd.Series(history['train_loss']).rolling(window=window).mean()
            ax.plot(epochs, moving_avg, 'r--', linewidth=1.5, alpha=0.7, label=f'Moving Average (window={window})')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        min_loss = min(history['train_loss'])
        min_epoch = history['train_loss'].index(min_loss) + 1
        ax.annotate(f'Min Loss: {min_loss:.4f}\nEpoch: {min_epoch}',
                   xy=(min_epoch, min_loss), xytext=(min_epoch + len(epochs)*0.1, min_loss + (max(history['train_loss']) - min_loss)*0.2),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        else:
            plt.savefig(self.output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_evaluation_metrics_dashboard(self, metrics: Dict, save_path: Optional[str] = None):
        """
        Create dashboard for evaluation metrics
        
        Args:
            metrics: Dictionary of evaluation metrics
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Silhouette Score
        ax1 = fig.add_subplot(gs[0, 0])
        if 'silhouette_score' in metrics and metrics['silhouette_score'] is not None:
            score = metrics['silhouette_score']
            ax1.barh(['Silhouette'], [score], color='steelblue' if score > 0 else 'coral')
            ax1.set_xlim(0, 1)
            ax1.set_xlabel('Score', fontsize=10)
            ax1.set_title('Silhouette Score', fontsize=11, fontweight='bold')
            ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            ax1.text(score + 0.05, 0, f'{score:.4f}', va='center', fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
            ax1.set_title('Silhouette Score', fontsize=11, fontweight='bold')
        
        # 2. Moran's I
        ax2 = fig.add_subplot(gs[0, 1])
        if 'morans_i' in metrics and metrics['morans_i'] is not None:
            moran_i = metrics['morans_i']
            colors = ['red' if moran_i < 0 else 'green' if moran_i > 0.3 else 'orange']
            ax2.barh(['Moran\'s I'], [moran_i], color=colors[0])
            ax2.set_xlim(-1, 1)
            ax2.set_xlabel('Score', fontsize=10)
            ax2.set_title('Moran\'s I (Spatial Autocorrelation)', fontsize=11, fontweight='bold')
            ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            ax2.text(moran_i + 0.05 if moran_i >= 0 else moran_i - 0.05, 0, 
                    f'{moran_i:.4f}', va='center', fontsize=12, fontweight='bold',
                    ha='left' if moran_i >= 0 else 'right')
        else:
            ax2.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
            ax2.set_title('Moran\'s I', fontsize=11, fontweight='bold')
        
        # 3. Coverage Metrics
        ax3 = fig.add_subplot(gs[0, 2])
        if 'coverage' in metrics:
            coverage = metrics['coverage']
            coverage_pct = coverage.get('coverage', 0) * 100
            ax3.pie([coverage_pct, 100 - coverage_pct], 
                   labels=[f'In Hotspots\n{coverage_pct:.1f}%', f'Outside\n{100-coverage_pct:.1f}%'],
                   autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#ecf0f1'])
            ax3.set_title('High-Severity Coverage', fontsize=11, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
            ax3.set_title('Coverage', fontsize=11, fontweight='bold')
        
        # 4. Hotspot Statistics
        ax4 = fig.add_subplot(gs[1, :])
        if 'n_hotspots' in metrics and 'n_spatial_units' in metrics:
            n_hotspots = metrics.get('n_hotspots', 0)
            n_units = metrics.get('n_spatial_units', 1)
            hotspot_ratio = metrics.get('hotspot_ratio', 0)
            avg_risk_hot = metrics.get('avg_risk_in_hotspots', 0)
            avg_risk_out = metrics.get('avg_risk_outside_hotspots', 0)
            
            categories = ['Hotspots', 'Non-Hotspots']
            counts = [n_hotspots, n_units - n_hotspots]
            risks = [avg_risk_hot, avg_risk_out]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax4_twin = ax4.twinx()
            bars1 = ax4.bar(x - width/2, counts, width, label='Count', color='steelblue', alpha=0.7)
            bars2 = ax4_twin.bar(x + width/2, risks, width, label='Avg Risk Score', color='coral', alpha=0.7)
            
            ax4.set_xlabel('Category', fontsize=11)
            ax4.set_ylabel('Count', fontsize=11, color='steelblue')
            ax4_twin.set_ylabel('Average Risk Score', fontsize=11, color='coral')
            ax4.set_xticks(x)
            ax4.set_xticklabels(categories)
            ax4.set_title('Hotspot vs Non-Hotspot Comparison', fontsize=12, fontweight='bold')
            
            # Add value labels
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                ax4.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + max(counts)*0.01,
                        f'{int(bar1.get_height())}', ha='center', va='bottom', fontsize=10)
                ax4_twin.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + max(risks)*0.01,
                             f'{bar2.get_height():.3f}', ha='center', va='bottom', fontsize=10, color='coral')
            
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')
        
        # 5. Risk Score Distribution
        ax5 = fig.add_subplot(gs[2, :2])
        if 'risk_scores' in metrics:
            risk_scores = metrics['risk_scores']
            ax5.hist(risk_scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax5.axvline(np.mean(risk_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(risk_scores):.3f}')
            ax5.axvline(np.median(risk_scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(risk_scores):.3f}')
            ax5.set_xlabel('Risk Score', fontsize=11)
            ax5.set_ylabel('Frequency', fontsize=11)
            ax5.set_title('Risk Score Distribution', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Risk scores not available', ha='center', va='center', fontsize=12)
            ax5.set_title('Risk Score Distribution', fontsize=12, fontweight='bold')
        
        # 6. Summary Statistics Table
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        summary_data = []
        if 'n_hotspots' in metrics:
            summary_data.append(['Hotspots Detected', f"{metrics.get('n_hotspots', 0):,}"])
        if 'hotspot_ratio' in metrics:
            summary_data.append(['Hotspot Ratio', f"{metrics.get('hotspot_ratio', 0):.2%}"])
        if 'silhouette_score' in metrics and metrics['silhouette_score'] is not None:
            summary_data.append(['Silhouette Score', f"{metrics['silhouette_score']:.4f}"])
        if 'morans_i' in metrics and metrics['morans_i'] is not None:
            summary_data.append(['Moran\'s I', f"{metrics['morans_i']:.4f}"])
        if 'coverage' in metrics:
            summary_data.append(['Coverage', f"{metrics['coverage'].get('coverage', 0):.2%}"])
        if 'avg_risk_in_hotspots' in metrics:
            summary_data.append(['Avg Risk (Hotspots)', f"{metrics.get('avg_risk_in_hotspots', 0):.4f}"])
        if 'avg_risk_outside_hotspots' in metrics:
            summary_data.append(['Avg Risk (Outside)', f"{metrics.get('avg_risk_outside_hotspots', 0):.4f}"])
        
        if summary_data:
            table = ax6.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                            cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            for i in range(len(summary_data) + 1):
                table[(i, 0)].set_facecolor('#ecf0f1')
                table[(i, 1)].set_facecolor('#ffffff')
            ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle('Model Evaluation Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Evaluation dashboard saved to {save_path}")
        else:
            plt.savefig(self.output_dir / 'evaluation_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self, history: Dict, metrics: Dict, 
                                    metadata: Optional[pd.DataFrame] = None,
                                    hotspot_results: Optional[Dict] = None):
        """
        Create interactive Plotly dashboard
        
        Args:
            history: Training history
            metrics: Evaluation metrics
            metadata: Original metadata
            hotspot_results: Hotspot detection results
        """
        if not HAS_PLOTLY:
            print("Plotly not available. Skipping interactive dashboard.")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Evaluation Metrics', 
                           'Hotspot Statistics', 'Risk Distribution'),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # 1. Training Loss
        if 'train_loss' in history and len(history['train_loss']) > 0:
            epochs = list(range(1, len(history['train_loss']) + 1))
            fig.add_trace(
                go.Scatter(x=epochs, y=history['train_loss'], 
                          mode='lines', name='Training Loss',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        # 2. Evaluation Metrics
        metric_names = []
        metric_values = []
        if 'silhouette_score' in metrics and metrics['silhouette_score'] is not None:
            metric_names.append('Silhouette')
            metric_values.append(metrics['silhouette_score'])
        if 'morans_i' in metrics and metrics['morans_i'] is not None:
            metric_names.append("Moran's I")
            metric_values.append(metrics['morans_i'])
        
        if metric_names:
            fig.add_trace(
                go.Bar(x=metric_names, y=metric_values, name='Metrics',
                      marker_color='steelblue'),
                row=1, col=2
            )
        
        # 3. Hotspot Statistics
        if 'n_hotspots' in metrics:
            fig.add_trace(
                go.Bar(x=['Hotspots', 'Non-Hotspots'],
                      y=[metrics.get('n_hotspots', 0), 
                        metrics.get('n_spatial_units', 0) - metrics.get('n_hotspots', 0)],
                      name='Count', marker_color='coral'),
                row=2, col=1
            )
        
        # 4. Risk Distribution (placeholder - would need actual risk scores)
        # This would be populated if risk_scores are available
        
        fig.update_layout(
            height=800,
            title_text="GNN Model Dashboard",
            showlegend=True
        )
        
        dashboard_path = self.output_dir / 'interactive_dashboard.html'
        fig.write_html(str(dashboard_path))
        print(f"Interactive dashboard saved to {dashboard_path}")
    
    def create_comprehensive_report(self, history: Dict, metrics: Dict, 
                                   results_file: Optional[str] = None):
        """
        Create comprehensive dashboard with all visualizations
        
        Args:
            history: Training history
            metrics: Evaluation metrics
            results_file: Optional path to results JSON file
        """
        print("Creating comprehensive dashboard...")
        
        # Load results if file provided
        if results_file:
            results = self.load_results(results_file)
            if results:
                if 'evaluation' in results:
                    metrics = results['evaluation']
                if 'num_epochs' in results:
                    # Reconstruct history if needed
                    if 'train_loss' not in history:
                        history = {'train_loss': []}
        
        # Create all dashboards
        self.create_training_curves(history)
        self.create_evaluation_metrics_dashboard(metrics)
        self.create_interactive_dashboard(history, metrics)
        
        print(f"\n All dashboards created in {self.output_dir}")
        print("   - training_curves.png")
        print("   - evaluation_dashboard.png")
        print("   - interactive_dashboard.html")

