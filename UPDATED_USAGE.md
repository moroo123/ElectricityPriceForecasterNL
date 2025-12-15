# Updated Feature Engine Usage

The feature engine has been enhanced with capacity factor features for solar and wind generation. These features normalize generation by the 99th percentile (capacity estimate) from the training data.

## New Features Added

1. **Residual load lags** (24h, 48h, 168h)
2. **Residual load rolling statistics** (mean, std, max, min over 24h)
3. **Renewable penetration lag** (24h)
4. **Peak hour indicators** (evening peak, morning peak, night, midday)
5. **Solar capacity factor** (solar / 99th percentile)
6. **Wind capacity factors** (onshore and offshore / 99th percentile)
7. **Wind volatility** (24h rolling std)
8. **Wind lags** (24h)
9. **Load ramping features** (1h and 24h changes, rolling max/min, above-average flag)

## Updated Usage Pattern

### Before (old way):
```python
# Create features
X_train, y_train = feature_engine.prepare_data(train_df, create_target=True)
X_val, y_val = feature_engine.prepare_data(val_df, create_target=True)
X_test, y_test = feature_engine.prepare_data(test_df, create_target=True)

# Fit scaler
feature_engine.fit(X_train, y_train)
```

### After (new way - with capacity estimates):
```python
# Create features
X_train, y_train = feature_engine.prepare_data(train_df, create_target=True)
X_val, y_val = feature_engine.prepare_data(val_df, create_target=True)
X_test, y_test = feature_engine.prepare_data(test_df, create_target=True)

# Fit scaler AND capacity estimates (pass raw training data)
feature_engine.fit(X_train, y_train, raw_data=train_df)

# Now transform
X_train_scaled = feature_engine.transform(X_train)
X_val_scaled = feature_engine.transform(X_val)
X_test_scaled = feature_engine.transform(X_test)
```

## Why This Change?

The capacity factors are now calculated using the 99th percentile of generation from the **training data only**. This prevents data leakage where future capacity information could influence past predictions.

The capacity estimates are fitted once on training data:
- `solar_capacity_` = 99th percentile of solar generation in training set
- `wind_onshore_capacity_` = 99th percentile of wind onshore in training set
- `wind_offshore_capacity_` = 99th percentile of wind offshore in training set

These estimates are then used consistently across train, validation, and test sets.

## Example Update for Notebooks

In `02_model_training.ipynb` and `03_hyperparameter_tuning.ipynb`, change:

```python
# OLD:
feature_engine.fit(X_train, y_train)

# NEW:
feature_engine.fit(X_train, y_train, raw_data=train_df)
```

That's it! The capacity factors will now be properly calculated and applied.

## Expected Impact

These new features should improve R² from ~0.46 to potentially 0.55-0.60+ because:

1. **Residual load lags** capture the key price driver (residual load has 0.354 correlation with price)
2. **Peak hour indicators** capture different pricing regimes during the day
3. **Capacity factors** normalize renewable generation, making it easier for the model to learn patterns
4. **Wind volatility** helps predict price volatility (wind uncertainty = price uncertainty)
5. **Load ramping** captures demand shocks that spike prices

After updating, you should see **80+ features** instead of the previous ~55.
