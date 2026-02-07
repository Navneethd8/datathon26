# How to Run Model Without Early Stopping & Update Results

## Step 1: Disable Early Stopping

Early stopping is already disabled in `config.yaml` (patience set to 999999). If you want to change it:

**Option A: Use current config (already disabled)**
- The config already has `early_stopping_patience: 999999` which effectively disables it

**Option B: Set to 0 to completely disable**
- Edit `GNN/config.yaml`:
  ```yaml
  early_stopping_patience: 0  # 0 = disabled
  ```

**Option C: Re-enable early stopping**
- Edit `GNN/config.yaml`:
  ```yaml
  early_stopping_patience: 10  # Will stop after 10 epochs without improvement
  ```

## Step 2: Run the Pipeline

Run the full pipeline with evaluation (required for automatic summary update):

```bash
cd GNN
python -m src.main --train --eval --visualize
```

This will:
- Train for all 100 epochs (no early stopping)
- Save results to `outputs/results.json`
- Generate visualizations

## Step 3: Update Results Summary

After the pipeline completes, automatically update the summary:

```bash
python update_summary.py
```

This script will:
- Load results from `outputs/results.json`
- Update `RESULTS_SUMMARY.md` with new metrics
- Preserve the document structure

## Manual Update (Alternative)

If you prefer to manually update, the results are saved in:
- `outputs/results.json` - Contains all metrics in JSON format
- Check this file for exact values to update in `RESULTS_SUMMARY.md`

## What Gets Updated Automatically

The `update_summary.py` script updates:
- Training epochs completed
- Final loss value
- Early stopping status
- Number of hotspots detected
- Evaluation metrics (Silhouette, Moran's I, Coverage)
- Timestamp

## Notes

- Make sure to run with `--eval` flag to generate `results.json`
- The summary update uses regex to find and replace values
- Original document structure and formatting is preserved
- If metrics aren't found, the script will warn you

