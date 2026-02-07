"""
Quick script to update RESULTS_SUMMARY.md after running the pipeline
Usage: python update_summary.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.update_results_summary import update_results_summary, load_metrics_from_log

if __name__ == '__main__':
    print("Loading results...")
    metrics = load_metrics_from_log()
    
    if metrics:
        print(f"Found results from {metrics.get('timestamp', 'unknown time')}")
        update_results_summary(metrics)
        print("\n Results summary updated successfully!")
    else:
        print("  No results.json found in outputs/")
        print("   Run the pipeline with --eval flag to generate results.json")
        print("   Or manually update RESULTS_SUMMARY.md")

