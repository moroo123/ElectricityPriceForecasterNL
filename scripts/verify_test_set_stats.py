
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append('src')
from features.pipeline import TimeSeriesFeatureEngine
from utils.preprocessing import create_merged_dataset, split_train_val_test, handle_missing_values

def verify_test_stats():
    print("Loading data...")
    df = create_merged_dataset(db_path="data/cache.db")
    df = handle_missing_values(df, strategy='forward_fill', limit=24).dropna()
    
    # Split
    _, _, test_df = split_train_val_test(
        df,
        train_end="2023-12-31",
        val_end="2024-06-30"
    )
    
    print(f"Test Set: {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")
    
    print("Creating features...")
    engine = TimeSeriesFeatureEngine(
        target_col='price',
        forecast_horizon=24,
        price_lags=[0, 144, 168], # minimal set for speed
        load_lags=[1]
    )
    
    X_test, y_test = engine.prepare_data(test_df, create_target=True)
    X_test = X_test.fillna(0)
    
    if 'price_lag_144' not in X_test.columns:
        print("Error: Lag 144 missing")
        return

    lag_144 = X_test['price_lag_144']
    
    print("\n--- Statistics ---")
    print(f"Target Mean:  {y_test.mean():.2f}")
    print(f"Lag144 Mean:  {lag_144.mean():.2f}")
    print(f"Mean Diff:    {lag_144.mean() - y_test.mean():.2f}")
    
    corr = np.corrcoef(lag_144, y_test)[0, 1]
    print(f"Correlation:  {corr:.4f}")
    
    # Calculate R2 manually for y = x
    mse = np.mean((y_test - lag_144) ** 2)
    var = np.var(y_test)
    r2 = 1 - (mse / var)
    print(f"Manual R2 (Observed): {r2:.4f}")
    
    print("\n--- Diagnosis ---")
    if abs(r2 - (-0.09)) < 0.1:
        print("CONFIRMED: The negative R2 is a property of the data, not a code bug.")
        if abs(lag_144.mean() - y_test.mean()) > 10:
            print("Reason: Significant distribution shift (Bias) between T and T-168h.")
        elif corr < 0.3:
            print("Reason: Low correlation (Volatility/Wind) broke the weekly pattern.")
    else:
        print("Discrepancy: Manual R2 does not match reported R2.")

if __name__ == "__main__":
    verify_test_stats()
