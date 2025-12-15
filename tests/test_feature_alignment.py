import sys
from pathlib import Path
import pandas as pd
import numpy as np
# import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.pipeline import TimeSeriesFeatureEngine

def test_future_covariates_alignment():
    """
    Verify that future covariates are correctly aligned.
    
    If we are at time t, attempting to predict price at t+24 (target),
    we should use load_forecast for t+24, because that forecast is available at time t.
    
    Currently (before fix), the pipeline likely uses load_forecast at t.
    """
    # Create dummy data
    dates = pd.date_range(start="2024-01-01", end="2024-01-05", freq="h")
    n = len(dates)
    
    df = pd.DataFrame({
        "price": np.arange(n),  # 0, 1, 2...
        "load_forecast": np.arange(n) * 10,  # 0, 10, 20...
        "actual_load": np.arange(n) * 10 + 5,
        "wind_onshore": np.zeros(n),
        "wind_offshore": np.zeros(n),
        "solar": np.zeros(n)
    }, index=dates)
    
    forecast_horizon = 24
    
    # Initialize engine
    # We expect to add a parameter like 'future_covariates' or similar logic
    engine = TimeSeriesFeatureEngine(
        target_col="price",
        forecast_horizon=forecast_horizon,
        feature_config={"scaler_type": None}, # Disable scaling for easy checking
        price_lags=[24], # Minimal lags
        load_lags=[] # Disable standard load lags for clarity
    )
    
    # We might need to configure future covariates if not default
    # For now, let's see what happens with default behavior
    try:
        # Check if attribute exists (it won't yet)
        engine.future_covariates = ["load_forecast"]
    except:
        pass

    X, y = engine.prepare_data(df)
    
    # Let's check a specific row.
    # We want to predict t+24.
    # Row index '2024-01-01 00:00:00' in X corresponds to prediction for '2024-01-02 00:00:00'.
    # Target y should be price['2024-01-02 00:00:00'].
    # Feature load_forecast should IDEALLY be load_forecast['2024-01-02 00:00:00'].
    
    test_idx = pd.Timestamp("2024-01-01 00:00:00")
    target_time = test_idx + pd.Timedelta(hours=forecast_horizon)
    
    print(f"\nTest time (t): {test_idx}")
    print(f"Target time (t+{forecast_horizon}): {target_time}")
    
    if test_idx not in X.index:
        print("Test index not in X (probably dropped due to lags)")
        # Find first available index
        test_idx = X.index[0]
        target_time = test_idx + pd.Timedelta(hours=forecast_horizon)
        print(f"New Test time (t): {test_idx}")
        print(f"New Target time (t+{forecast_horizon}): {target_time}")

    # Check target alignment
    target_val = y.loc[test_idx]
    expected_target = df.loc[target_time, "price"]
    assert target_val == expected_target, f"Target alignment incorrect. Got {target_val}, expected {expected_target}"
    print("Target alignment verified.")
    
    # Check simple load_forecast feature
    # NOTE: The current column name might be just 'load_forecast'
    col_name = "load_forecast" 
    
    # Check if we have the future aligned feature
    future_col_name = "load_forecast_future"
    
    if future_col_name in X.columns:
        print(f"Found future column: {future_col_name}")
        val = X.loc[test_idx, future_col_name]
        expected_val = df.loc[target_time, "load_forecast"] # Expected: 240
        
        print(f"Value at t: {val}")
        print(f"Value at t+{forecast_horizon} (expected): {expected_val}")
        
        if val == expected_val:
            print("SUCCESS: Future covariate is aligned!")
        else:
            print("FAILURE: Future covariate is NOT aligned (likely using current value).")
            # Fail the test if we expect it to work
            assert val == expected_val, f"Expected {future_col_name} to be {expected_val}, got {val}"
            
    elif col_name in X.columns:
        print(f"Checking existing column: {col_name}")
        val = X.loc[test_idx, col_name]
        expected_val = df.loc[target_time, "load_forecast"]
        current_val = df.loc[test_idx, "load_forecast"]
        
        print(f"Value at X[t]: {val}")
        print(f"Value at df[t+{forecast_horizon}] (Target Time): {expected_val}")
        print(f"Value at df[t] (Decision Time): {current_val}")
        
        if val == expected_val:
             print("SUCCESS: Feature is aligned with target time!")
        elif val == current_val:
             print("FAILURE: Feature is using current time (t), not target time (t+24).")
             # We want this to eventually pass
             pytest.fail("Feature is using current time instead of target time")
        else:
             print(f"FAILURE: Feature value {val} matches neither current {current_val} nor target {expected_val}")

if __name__ == "__main__":
    test_future_covariates_alignment()
