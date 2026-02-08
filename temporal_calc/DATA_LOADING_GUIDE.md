# Dashboard Data Loading Guide

The dashboard can now load data from multiple sources in the `outputs/` directory.

## How It Works

The dashboard uses JavaScript `fetch()` to load data files when the page loads. There are two approaches:

### 1. **JSON Files** (Recommended)
JSON files are loaded directly and parsed automatically:
- `outputs/map_data.json` - Map visualization data
- `outputs/model_results.json` - Model performance metrics

### 2. **CSV Files** (Automatic Parsing)
CSV files are automatically parsed using a built-in CSV parser:
- `outputs/neighborhood_risk_forecast.csv` - Neighborhood predictions
- `outputs/accessibility_scores.csv` - Accessibility scores by neighborhood
- `outputs/eval/metrics.csv` - Evaluation metrics (MAE, RMSE, etc.)

## Current Data Loading

The dashboard automatically loads:

1. **Map Data** (`outputs/map_data.json`)
   - Used for interactive map visualization
   - Contains barrier points, grid cells, neighborhoods
   - Updates statistics cards

2. **Model Metrics** (`outputs/model_results.json`)
   - Updates the model performance comparison table
   - Shows MAE, R², accuracy metrics for Transformer, LSTM, XGBoost

3. **Evaluation Metrics** (`outputs/eval/metrics.csv`)
   - Loaded but not yet displayed (available in console)
   - Contains MAE, RMSE, MAPE, Correlation

4. **Neighborhood Forecast** (`outputs/neighborhood_risk_forecast.csv`)
   - Loaded but predictions table uses map_data.json
   - Available for future enhancements

5. **Accessibility Scores** (`outputs/accessibility_scores.csv`)
   - Loaded and available for use
   - Contains neighborhood accessibility rankings

## Adding New Data

To add new data to the dashboard:

### Option 1: Add to Existing Functions

```javascript
// In dashboard.html, add a new function:
async function loadMyNewData() {
    try {
        const data = await loadCSV('outputs/my_data.csv');
        // Use the data to update the dashboard
        console.log('Loaded:', data);
    } catch (error) {
        console.warn('Data not found');
    }
}

// Call it in loadData():
async function loadData() {
    // ... existing loads ...
    loadMyNewData();
}
```

### Option 2: Convert CSV to JSON (More Reliable)

For complex CSV files, convert them to JSON first:

```bash
python convert_csv_to_json.py
```

This creates JSON versions of CSV files that are easier to load and parse.

## Using the Data

Once loaded, data is available in JavaScript:

```javascript
// CSV data is parsed into arrays of objects
const forecastData = await loadCSV('outputs/neighborhood_risk_forecast.csv');
// forecastData[0] = { neighborhood: "...", current_score: 105.39, ... }

// JSON data is already parsed
const modelMetrics = await fetch('outputs/model_results.json').then(r => r.json());
// modelMetrics.Transformer.mae = 16.01
```

## Troubleshooting

### CORS Errors
If you see CORS errors when opening the HTML file directly:
- **Solution**: Serve the dashboard through a local web server:
  ```bash
  # Python 3
  python -m http.server 8000
  
  # Then open: http://localhost:8000/dashboard.html
  ```

### CSV Parsing Issues
If CSV files have complex formatting (quoted fields, commas inside quotes):
- **Solution**: Use `convert_csv_to_json.py` to convert to JSON first

### Missing Data
If data files are missing:
- Check that the pipeline has been run: `python run_all.sh` or `python run_pipeline.py`
- Verify files exist in `outputs/` directory
- Check browser console for error messages

## File Structure

```
temporal_calc/
├── dashboard.html          # Main dashboard (loads data from outputs/)
├── outputs/
│   ├── map_data.json       # Loaded automatically
│   ├── model_results.json  # Loaded automatically
│   ├── neighborhood_risk_forecast.csv  # Loaded automatically
│   ├── accessibility_scores.csv        # Loaded automatically
│   ├── eval/
│   │   └── metrics.csv     # Loaded automatically
│   └── plots/              # Images referenced in HTML
└── convert_csv_to_json.py  # Optional: Convert CSV to JSON
```

## Example: Displaying Evaluation Metrics

To display evaluation metrics in the dashboard, you could add:

```javascript
async function displayEvaluationMetrics() {
    const data = await loadCSV('outputs/eval/metrics.csv');
    if (data.length > 0) {
        const metrics = data[0];
        // Update a display element
        document.getElementById('eval-mae').textContent = metrics.MAE.toFixed(2);
        document.getElementById('eval-rmse').textContent = metrics.RMSE.toFixed(2);
        document.getElementById('eval-correlation').textContent = metrics.Correlation.toFixed(3);
    }
}
```

Then call it in `loadData()`.

