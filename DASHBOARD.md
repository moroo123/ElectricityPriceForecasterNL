# Electricity Price Forecasting Dashboard

A Streamlit-based interactive dashboard for comparing different electricity price forecasting models.

## Features

- **Model Comparison**: Compare multiple models side-by-side (XGBoost models and baseline models)
- **Interactive Visualizations**:
  - Time series predictions comparison
  - Performance metrics bar charts
  - Scatter plots (predicted vs actual)
  - Error distribution histograms
  - Error analysis by hour of day
  - Feature importance charts
- **Flexible Dataset Selection**: View results on train, validation, or test sets
- **Model Selection**: Enable/disable specific models to compare
- **Configurable Time Range**: Adjust the number of hours to display

## Installation

1. Install Streamlit (if not already installed):
```bash
pip install -r requirements.txt
```

## Running the Dashboard

From the project root directory, run:

```bash
streamlit run app.py
```

The dashboard will open automatically in your default web browser (usually at http://localhost:8501).

## How to Use

### 1. Select Dataset
Use the sidebar to choose which dataset to analyze:
- **Test Set**: Most recent data (2024-06-30 to 2025-01-01)
- **Validation Set**: Middle period (2024-01-01 to 2024-06-30)
- **Train Set**: Historical data (2022-01-01 to 2023-12-31)

### 2. Select Models
Check/uncheck models to include in the comparison:
- **Baseline 24h Persistence**: Simple model that predicts tomorrow's price = today's price
- **Baseline 168h Persistence**: Predicts next week's price = last week's price
- **XGBoost Models**: All trained XGBoost models from the `models/` directory

### 3. Adjust Visualization
- Use the **Hours to display** slider to control the time window shown in predictions plot
- Default is 7 days (168 hours), but you can view from 24 hours to full week

### 4. Explore Results

The dashboard shows:

#### Performance Metrics
- Summary cards showing best MAE, RMSE, and R² across all selected models
- Table with detailed metrics for each model
- Bar chart comparing MAE and RMSE

#### Predictions Comparison
- Interactive time series plot showing actual prices vs predictions from all models
- Hover over the plot to see exact values
- Toggle models on/off by clicking legend items

#### Detailed Model Analysis
- Select any model for deeper analysis
- **Scatter plot**: Shows how well predictions match actual values
- **Error distribution**: Histogram showing prediction error patterns
- **Hourly error analysis**: See which hours of day are hardest to predict

#### Feature Importance (XGBoost only)
- Top 20 most important features for the selected XGBoost model
- Helps understand what drives the model's predictions

## Available Models

The dashboard automatically detects all models in the `models/` directory. Currently available:

1. **xgboost_24h_forecast**: Basic XGBoost model with default hyperparameters
2. **xgboost_24h_tuned**: Hyperparameter-optimized XGBoost model (via Optuna)

## Tips

- Start with the **Test Set** to see final model performance
- Compare against **Baseline 24h** to see if ML models add value (they should have lower MAE/RMSE)
- Look for patterns in the **Error by Hour** chart - are certain times harder to predict?
- Check **Feature Importance** to understand what drives predictions (usually recent price lags)
- Use the **Scatter Plot** to identify systematic biases (points should scatter around the diagonal line)

## Troubleshooting

### Models not showing up
- Make sure you've trained models first using the notebooks
- Check that `.json` model files and `_feature_engine.pkl` files exist in `models/` directory

### Data not loading
- Ensure `data/cache.db` exists and contains data
- Run the data download notebook first if needed

### Streamlit errors
- Make sure you're running from the project root directory
- Check that all dependencies are installed: `pip install -r requirements.txt`
