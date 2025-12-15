# Performance Improvements Guide

## Current Status
- **Current R²**: 0.44 (test set)
- **Target R²**: 0.55-0.65 (realistic ceiling given 0.354 max correlation)

## Recent Improvements Added

### 1. Enhanced Residual Load Features (15+ new features)

Residual load (load - renewables) is your strongest predictor with 0.354 correlation. Added:

- **Volatility features**: 24h and 168h rolling std (uncertainty drives price spikes)
- **Rapid changes**: 1h and 24h changes, absolute changes (ramps affect pricing)
- **Percentile ranking**: Relative position within 24h and 168h windows
- **Extreme indicators**: Binary flags for high/very high/low residual load states

**Why this helps**: Residual load drives prices, but **how it changes** matters more than absolute values. Volatility and rapid changes capture supply-demand imbalances.

### 2. Price Regime Features (4 new features)

Electricity markets have distinct pricing regimes with different dynamics:

- **Negative prices** (< 0): Renewable oversupply, generators pay to offload
- **Scarcity prices** (> 500): Generation shortage, extreme prices
- **Extreme scarcity** (> 700): Critical shortage
- **Normal regime** (0-500): Regular supply-demand balance

**Why this helps**: The relationship between features and price changes dramatically in different regimes. Binary flags let the model learn separate patterns.

### 3. Capacity Factor Features (3 new features)

- Solar capacity factor (normalized by 99th percentile)
- Wind onshore capacity factor
- Wind offshore capacity factor

**Why this helps**: Raw generation values depend on installed capacity. Normalizing shows **utilization**, which is more predictive.

## How to Use

### Step 1: Restart Notebook Kernel
The module has been updated, so restart to reload it.

### Step 2: Update Training Code

**Only ONE line needs to change:**

```python
# Before:
X_train, y_train = feature_engine.prepare_data(train_df, create_target=True)

# After (add fit_capacity=True):
X_train, y_train = feature_engine.prepare_data(train_df, create_target=True, fit_capacity=True)
```

Everything else stays the same.

### Step 3: Check Feature Count

You should now see **100+ features** instead of ~55. Run:

```python
print(f"Total features: {X_train.shape[1]}")
```

### Step 4: Retrain and Compare

After retraining, check if R² improves. Expected:
- **Baseline**: 0.44
- **With new features**: 0.50-0.55
- **With hyperparameter tuning**: 0.55-0.60

## Additional Recommendations

### A. Handle NaN Values Properly

**Current approach** (in notebooks): `X_train.fillna(0)`

**Problem**: Zero is not a valid price/load value. This creates fake training data.

**Better approach**:

```python
# Option 1: Drop rows with NaN (safest)
# After prepare_data, before fitting:
mask = ~X_train.isna().any(axis=1)
X_train_clean = X_train[mask]
y_train_clean = y_train[mask]

# Option 2: Forward fill (for time series)
X_train_clean = X_train.fillna(method='ffill')

# Then proceed with fitting
feature_engine.fit(X_train_clean, y_train_clean)
```

### B. Adjust Hyperparameters

With more features and weak signals, prevent overfitting:

```python
params = {
    'learning_rate': 0.03,     # Slower learning
    'max_depth': 4,            # Shallower trees (was 6-8)
    'min_child_weight': 5,     # More conservative (was 1-3)
    'subsample': 0.8,
    'colsample_bytree': 0.7,   # Use fewer features per tree
    'reg_alpha': 0.1,          # L1 regularization
    'reg_lambda': 2.0,         # Stronger L2 regularization
}
```

### C. Feature Importance Analysis

After training, check which features matter:

```python
importance = get_feature_importance(model, X_train.columns.tolist(), top_n=30)
print(importance)
```

Look for:
- Are residual load features in top 10?
- Are price regime features being used?
- Are capacity factors helping?

If not, the model might not be finding the signal. Try deeper trees (max_depth=6) or less regularization.

## Why R² = 0.65 is Likely the Ceiling

Your data exploration showed:
- **Residual load correlation**: 0.354 (explains 12.5% of variance)
- **Total renewable correlation**: -0.329 (explains 10.8% of variance)
- **Load forecast correlation**: 0.236 (explains 5.6% of variance)

Even combining all features optimally, you're limited by the fundamental signal strength. To break 0.65, you'd need:

1. **External data**: Weather forecasts, fuel prices, cross-border flows
2. **Market data**: Bid-ask spreads, volume, interconnector utilization
3. **Supply data**: Planned outages, maintenance schedules
4. **Different modeling**: Quantile regression, probabilistic forecasting, regime-switching models

## Troubleshooting

### "Feature count didn't increase"
- Restart notebook kernel
- Check that `fit_capacity=True` is on the training set line
- Verify the module reloaded: `import importlib; importlib.reload(pipeline)`

### "Performance got worse"
- Check for inf/NaN values: `X_train.replace([np.inf, -np.inf], np.nan).isna().sum()`
- The percentile ranking features might create NaN on short windows
- Try dropping rows with NaN instead of filling with 0

### "Training is very slow"
- Percentile ranking features use `rolling().apply()` which is slow
- On 26k+ samples, this might take 1-2 minutes
- This is normal - only runs once during feature creation

## Expected Results

After implementing these changes:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Features | ~55 | 100+ |
| Train R² | 0.96 (overfitting) | 0.85-0.90 |
| Val R² | 0.44 | 0.50-0.55 |
| Test R² | 0.44 | 0.50-0.60 |
| Test MAE | 27 €/MWh | 22-25 €/MWh |

The key improvement is more stable performance across all sets (less overfitting).
