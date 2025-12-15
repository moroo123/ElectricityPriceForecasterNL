
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append('src')
from features.pipeline import TimeSeriesFeatureEngine
from utils.preprocessing import create_merged_dataset, handle_missing_values

def verify_lags():
    print("Loading data...")
    df = create_merged_dataset(db_path="data/cache.db")
    df = handle_missing_values(df, strategy='forward_fill', limit=24).dropna()
    
    print("Creating features...")
    # Matches notebook config exactly
    engine = TimeSeriesFeatureEngine(
        target_col='price',
        forecast_horizon=24,
        price_lags=[0, 1, 2, 3, 6, 12, 24, 48, 144, 168],
        load_lags=[1, 24, 168]
    )
    
    # We only care about X vs y alignment
    X, y = engine.prepare_data(df)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    if 'price_lag_144' not in X.columns:
        print("ERROR: price_lag_144 NOT FOUND in columns!")
        print(X.columns)
        return

    # Check for zeros (potential fillna issue)
    zeros = (X['price_lag_144'] == 0).sum()
    print(f"price_lag_144 zeros: {zeros} / {len(X)}")
    
    # Check correlation
    # We expect high correlation between X['price_lag_144'] and y
    # y[t] is Price[t+24]
    # X['price_lag_144'][t] is Price[t-144]
    # Time diff: 168 hours (1 week). Expect strong correlation.
    
    corr = np.corrcoef(X['price_lag_144'], y)[0, 1]
    print(f"Correlation (Lag 144 vs Target): {corr:.4f}")
    
    # Check Lag 0 vs Target (24h correlation)
    if 'price_lag_0' in X.columns:
        corr0 = np.corrcoef(X['price_lag_0'], y)[0, 1]
        print(f"Correlation (Lag 0 vs Target): {corr0:.4f}")
        
    # Validation on specific dates (Winter vs Summer)
    # Print a few samples
    print("\nSample Data (Tail):")
    sample = pd.DataFrame({
        'Target (T+24)': y.tail(),
        'Lag 144 (T-144)': X['price_lag_144'].tail(),
        'Lag 0 (T)': X['price_lag_0'].tail()
    })
    print(sample)

if __name__ == "__main__":
    verify_lags()
