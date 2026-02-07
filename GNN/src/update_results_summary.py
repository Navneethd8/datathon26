"""
Script to automatically update RESULTS_SUMMARY.md based on pipeline execution
Run this after running the main pipeline to update the summary with new results
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
import re


def load_metrics_from_log(log_file_path: str = None) -> dict:
    """
    Try to extract metrics from log or use default structure
    For now, we'll create a template that can be filled manually or via JSON
    """
    # If there's a results JSON file, load it
    results_file = Path(__file__).parent.parent / 'outputs' / 'results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}


def update_results_summary(metrics: dict = None):
    """
    Update RESULTS_SUMMARY.md with new metrics
    """
    summary_path = Path(__file__).parent.parent / 'RESULTS_SUMMARY.md'
    
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found!")
        return
    
    with open(summary_path, 'r') as f:
        content = f.read()
    
    # Update timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = re.sub(
        r'## Pipeline Execution Summary\n\nThe GNN pipeline successfully completed',
        f'## Pipeline Execution Summary\n\n**Last Updated**: {timestamp}\n\nThe GNN pipeline successfully completed',
        content
    )
    
    # Update training epochs if provided
    if metrics and 'num_epochs' in metrics:
        content = re.sub(
            r'Trained for \d+ epochs',
            f'Trained for {metrics["num_epochs"]} epochs',
            content
        )
        content = re.sub(
            r'early stopping triggered',
            'training completed' if metrics.get('early_stopped', False) == False else 'early stopping triggered',
            content
        )
    
    # Update final loss if provided
    if metrics and 'final_loss' in metrics:
        content = re.sub(
            r'Final loss: \d+\.\d+',
            f'Final loss: {metrics["final_loss"]:.4f}',
            content
        )
    
    # Update hotspot count if provided
    if metrics and 'n_hotspots' in metrics:
        content = re.sub(
            r'Detected \d+ hotspots',
            f'Detected {metrics["n_hotspots"]} hotspots',
            content
        )
        content = re.sub(
            r'\*\*Number of hotspots\*\*: \d+',
            f'**Number of hotspots**: {metrics["n_hotspots"]}',
            content
        )
    
    # Update evaluation metrics if provided
    if metrics and 'evaluation' in metrics:
        eval_metrics = metrics['evaluation']
        
        if 'silhouette_score' in eval_metrics:
            content = re.sub(
                r'Silhouette Score: \d+\.\d+',
                f'Silhouette Score: {eval_metrics["silhouette_score"]:.4f}',
                content
            )
        
        if 'morans_i' in eval_metrics and eval_metrics['morans_i'] is not None:
            content = re.sub(
                r"Moran's I: \d+\.\d+",
                f"Moran's I: {eval_metrics['morans_i']:.4f}",
                content
            )
        
        if 'coverage' in eval_metrics:
            coverage = eval_metrics['coverage']
            if 'coverage' in coverage:
                content = re.sub(
                    r'Coverage: \d+\.\d+%',
                    f'Coverage: {coverage["coverage"]:.2%}',
                    content
                )
    
    # Write updated content
    with open(summary_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {summary_path}")


if __name__ == '__main__':
    metrics = load_metrics_from_log()
    update_results_summary(metrics)
    print("Results summary updated!")

