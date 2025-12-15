
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to python path if not already there
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.pipeline import create_feature_pipeline, TimeSeriesFeatureEngine

def test_feature_pipeline_allows_recent_lags():
    """Test that the pipeline allows lags smaller than forecast horizon."""
    # Create simple dummy data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "price": np.arange(100, dtype=float),
        "load_forecast": np.random.randn(100)
    }, index=dates)

    # Config with lags < horizon
    # Horizon 24, but we want Lag 0 (current price) to be allowed
    try:
        pipeline = create_feature_pipeline(
            forecast_horizon=24,
            price_lags=[0, 1, 24],
            load_lags=[0, 24]
        )
        
        # Fit transform
        X = pipeline.fit_transform(df)
        
        # Check if columns exist
        assert "price_lag_0" in X.columns
        assert "price_lag_1" in X.columns
        assert "price_lag_24" in X.columns
        
        # Verify values
        # price_lag_0 at index i should be price[i]
        # (Since this is just feature creation, not data preparation with shift)
        # Note: pipeline creates features at row T based on T. 
        # prepare_data handles the target shifting.
        
        assert X["price_lag_0"].iloc[50] == df["price"].iloc[50]
        assert X["price_lag_1"].iloc[50] == df["price"].iloc[49]
        
    except ValueError as e:
        pytest.fail(f"Pipeline raised ValueError for recent lags: {e}")

def test_time_series_engine_feature_alignment():
    """
    Test that TimeSeriesFeatureEngine aligns features correctly.
    X[t] should contain Price[t] (Lag 0) and Target should be Price[t+24].
    """
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "price": np.arange(100, dtype=float)
    }, index=dates)
    
    engine = TimeSeriesFeatureEngine(
        target_col="price",
        forecast_horizon=24,
        price_lags=[0, 24]
    )
    
    X, y = engine.prepare_data(df)
    
    # We lose the last 24 points for y because they would be in future
    # And we assume we lose some initial points due to lags/shifting if any (Lag 24)
    # y[t] corresponds to price[t+24]
    
    # Pick a valid index in the middle
    # X index 2024-01-02 00:00:00 (Row 24)
    # Target should be 2024-01-03 00:00:00 (Row 48) => Price = 48
    
    test_idx = dates[24] # "2024-01-02 00:00:00"
    
    if test_idx in X.index:
        # Feature at T
        x_row = X.loc[test_idx]
        # Lag 0 should be Price[T] = 24
        assert x_row["price_lag_0"] == 24.0
        
        # Target at T should be Price[T+24] = 48
        # y is a Series aligned with X index
        y_val = y.loc[test_idx]
        assert y_val == 48.0
        
    else:
        pytest.fail("Test index not found in result DataFrame")

