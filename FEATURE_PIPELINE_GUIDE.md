# Feature Pipeline Guide

This guide explains how to use the feature pipeline for electricity price forecasting.

## Overview

The feature pipeline is a comprehensive system for creating features from electricity market data and training machine learning models (XGBoost, etc.) for price forecasting.

## Architecture

```
src/
├── features/
│   ├── temporal.py          # Time-based features (hour, day, month, etc.)
│   ├── lag_features.py      # Lag and rolling window features
│   ├── market_features.py   # Market-specific features (residual load, etc.)
│   └── pipeline.py          # Main feature pipeline
├── models/
│   └── train.py            # Model training and evaluation
└── utils/
    └── preprocessing.py    # Data loading and preprocessing
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Load Data

```python
from src.utils.preprocessing import create_merged_dataset, split_train_val_test

# Load all data from cache
df = create_merged_dataset(db_path="data/cache.db")

# Split into train/val/test (time-based, no shuffling!)
train_df, val_df, test_df = split_train_val_test(
    df,
    train_end="2023-12-31",
    val_end="2024-06-30"
)
```

### 3. Create Features

```python
from src.features.pipeline import TimeSeriesFeatureEngine

# Initialize feature engine
feature_engine = TimeSeriesFeatureEngine(
    target_col='price',
    forecast_horizon=24,  # Predict 24 hours ahead
    feature_config={'scaler_type': 'robust'}
)

# Create features and target
X_train, y_train = feature_engine.prepare_data(train_df, create_target=True)
X_val, y_val = feature_engine.prepare_data(val_df, create_target=True)
X_test, y_test = feature_engine.prepare_data(test_df, create_target=True)
```

### 4. Train Model

```python
from src.models.train import train_xgboost_model, evaluate_model

# Fit scaler and transform features
feature_engine.fit(X_train, y_train)
X_train_scaled = feature_engine.transform(X_train)
X_val_scaled = feature_engine.transform(X_val)
X_test_scaled = feature_engine.transform(X_test)

# Fill any NaN values
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index).fillna(0)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index).fillna(0)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index).fillna(0)

# Train XGBoost
model = train_xgboost_model(
    X_train_scaled,
    y_train,
    X_val_scaled,
    y_val,
    n_estimators=1000,
    early_stopping_rounds=50
)

# Evaluate
test_metrics = evaluate_model(model, X_test_scaled, y_test, set_name="Test")
```

### 5. Save and Load Model

```python
from src.models.train import save_model, load_model

# Save
save_model(model, feature_engine, save_dir="models", model_name="my_model")

# Load
model, feature_engine = load_model(save_dir="models", model_name="my_model")
```

## Features Created

### 1. Temporal Features
- **Hour of day** (cyclical encoding with sin/cos)
- **Day of week** (cyclical encoding)
- **Month** (cyclical encoding)
- **Weekend flag**
- **Dutch holidays** (optional, requires `holidays` package)

### 2. Market Features
- **Residual load** = Load - Total Renewable Generation
  - This is a key driver (correlation: 0.354 with price)
- **Total renewable generation** = Wind Onshore + Wind Offshore + Solar
- **Total wind** = Wind Onshore + Wind Offshore
- **Renewable penetration ratio** = (Total Renewable / Load) × 100
- **Individual renewable ratios** (wind/load, solar/load, etc.)
- **Interaction features**:
  - Residual load × hour
  - Solar × hour (captures diurnal pattern)
  - Wind × hour

### 3. Lag Features
- **Price lags**: t-24, t-48, t-168 hours (1 day, 2 days, 1 week)
- **Load lags**: t-1, t-24, t-168 hours
- **Residual load lags**: t-1, t-24, t-168 hours

### 4. Rolling Window Features
- **Rolling statistics** (24h, 48h, 168h windows):
  - Mean, std, min, max
  - Applied to price, load, residual load

## Important: Avoiding Data Leakage

The pipeline is designed to prevent data leakage:

1. **Time-based splitting**: Always use `split_train_val_test()` which splits chronologically
2. **Forecast horizon**: Target is shifted by `forecast_horizon` hours
3. **Lag features**: All price lags are ≥ forecast_horizon
4. **No future information**: Features only use past data

## Customization

### Custom Feature Pipeline

```python
from src.features.temporal import TemporalFeatures
from src.features.market_features import MarketFeatures
from src.features.lag_features import LagFeatures

# Create custom transformers
temporal = TemporalFeatures(
    features=['hour', 'day_of_week', 'month'],
    cyclical=True
)

market = MarketFeatures(
    create_residual_load=True,
    create_renewable_ratio=True
)

lags = LagFeatures(
    lags=[24, 48, 168],
    columns=['price'],
    fill_value=0
)

# Apply transformers
temporal_features = temporal.fit_transform(df)
market_features = market.fit_transform(df)
lag_features = lags.fit_transform(df)

# Combine
features = pd.concat([temporal_features, market_features, lag_features], axis=1)
```

### Custom Model Parameters

```python
# Customize XGBoost hyperparameters
custom_params = {
    'learning_rate': 0.03,
    'max_depth': 8,
    'min_child_weight': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'gamma': 0.2,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
}

model = train_xgboost_model(
    X_train, y_train,
    X_val, y_val,
    params=custom_params
)
```

## Example Notebook

See [notebooks/02_model_training.ipynb](notebooks/02_model_training.ipynb) for a complete example with:
- Data loading and preprocessing
- Feature engineering
- Model training
- Evaluation and visualization
- Feature importance analysis

## Performance Tips

1. **Missing values**: Handle with `handle_missing_values()` before feature creation
2. **Outliers**: Price data has outliers (negative prices, extreme spikes). RobustScaler handles these well.
3. **Feature selection**: Use `get_feature_importance()` to identify top features
4. **Cross-validation**: Use `cross_validate_model()` for robust performance estimates
5. **Hyperparameter tuning**: Use validation set for early stopping
6. **Lag features**: For 24h forecasting, use same-hour lags (24, 48, 72... 168h) - these predict the same hour on different days
7. **Feature engineering**: Add price differences, volatility, and renewable penetration flags for better performance
8. **Model complexity**: Don't over-regularize - electricity prices have complex patterns that need deeper trees (max_depth=8+)

## Evaluation Metrics

The pipeline provides these metrics:
- **MAE** (Mean Absolute Error): Average prediction error in €/MWh
- **RMSE** (Root Mean Squared Error): Penalizes large errors more
- **MAPE** (Mean Absolute Percentage Error): Relative error as percentage
- **R²** (R-squared): Proportion of variance explained

Based on your data:
- Average price: 138.28 €/MWh
- Std dev: 113.14 €/MWh
- Target MAE: < 15 €/MWh would be good performance

## Data Requirements

The pipeline expects these columns in the input DataFrame:
- `price`: Day-ahead electricity price (€/MWh)
- `load_forecast`: Load forecast (MW)
- `actual_load`: Actual load (MW) - optional
- `wind_onshore`: Wind onshore generation (MW)
- `wind_offshore`: Wind offshore generation (MW)
- `solar`: Solar generation (MW)

The DataFrame must have a `DatetimeIndex` in hourly frequency.

## Troubleshooting

### Issue: "DataFrame must have DatetimeIndex"
**Solution**: Ensure your DataFrame has a proper DatetimeIndex:
```python
df.index = pd.to_datetime(df.index)
```

### Issue: NaN values in features
**Solution**: Fill NaN values before or after feature creation:
```python
X = X.fillna(0)  # or use handle_missing_values()
```

### Issue: Model not converging
**Solution**:
- Check for inf values: `X.replace([np.inf, -np.inf], np.nan)`
- Increase `n_estimators`
- Adjust `learning_rate`
- Check feature scales (use RobustScaler for outliers)

## Next Steps

1. Run [notebooks/02_model_training.ipynb](notebooks/02_model_training.ipynb)
2. Experiment with different forecast horizons (6h, 12h, 24h, 48h)
3. Try other models (LightGBM, CatBoost, Random Forest)
4. Add external features (temperature, cross-border flows, fuel prices)
5. Implement online learning for model updates
6. Deploy model to production

## References

- Your data exploration: [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)
- Key finding: Residual load has 0.354 correlation with price
- Data source: ENTSO-E Transparency Platform
