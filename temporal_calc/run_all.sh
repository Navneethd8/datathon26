#!/bin/bash

# Datathon 26: Seattle Accessibility Analysis - End-to-End Pipeline
# Automates: EDA -> Feature Engineering -> Model Training -> Map Generation

set -e # Exit on error

# Configuration
PROJECT_DIR="/Volumes/NavDisk/datathon26/temporal_calc"
OUTPUT_DIR="$PROJECT_DIR/outputs"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "=========================================================="
echo "STARTING END-TO-END ACCESSIBILITY PIPELINE"
echo "=========================================================="
date

cd "$PROJECT_DIR"

# 1. Exploratory Data Analysis
echo -e "\n[1/4] Running Exploratory Data Analysis (eda.py)..."
python3 eda.py > "$LOG_DIR/eda.log" 2>&1
echo "EDA Complete. Plots saved to $OUTPUT_DIR/plots/"

# 2. Feature Engineering
echo -e "\n[2/4] Running Feature Engineering (feature_engineering.py)..."
python3 feature_engineering.py > "$LOG_DIR/feature_engineering.log" 2>&1
echo "Feature Engineering Complete. Data saved to $OUTPUT_DIR/processed_timeseries.csv"

# 3. Model Training and Orchestration
echo -e "\n[3/4] Running Model Orchestrator (max_performance.py)..."
echo "   (This may take several minutes for Transformer/LSTM training)"
python3 max_performance.py > "$LOG_DIR/max_performance.log" 2>&1
echo "Model Training Complete. Results saved to $OUTPUT_DIR/model_results.json"

# 4. Map Data Generation
echo -e "\n[4/4] Generating Map Data (generate_map_data.py)..."
python3 generate_map_data.py > "$LOG_DIR/generate_map_data.log" 2>&1
echo "Map Data Generated. Saved to $OUTPUT_DIR/map_data.json"

echo -e "\n=========================================================="
echo "PIPELINE SUCCESSFUL!"
echo "=========================================================="
echo "You can now view the results by opening:"
echo "  $PROJECT_DIR/dashboard.html"
echo "=========================================================="
date
