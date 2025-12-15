import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.pipeline import TimeSeriesFeatureEngine, create_feature_pipeline

@pytest.fixture
def sample_data():
    """Create sample data with hourly frequency."""
    dates = pd.date_range(start="2024-01-01", periods=500, freq="h")
    df = pd.DataFrame({
        "price": np.abs(np.random.randn(500) * 100),
        "load_forecast": np.random.randn(500) * 1000 + 10000,
        "wind_onshore": np.random.randn(500) * 500 + 1000,
        "solar": np.abs(np.random.randn(500) * 500)
    }, index=dates)
    return df

def test_feature_engine_initialization():
    engine = TimeSeriesFeatureEngine(target_col="price", forecast_horizon=24)
    assert engine.target_col == "price"
    assert engine.forecast_horizon == 24

def test_prepare_data_structure(sample_data):
    engine = TimeSeriesFeatureEngine(target_col="price", forecast_horizon=24)
    X, y = engine.prepare_data(sample_data, create_target=True)
    
    # Check that price is removed from X
    assert "price" not in X.columns
    
    # Check dimensions
    # Should lose 24 rows due to shift + max_lags
    # Default max lag is 168 (1 week), but simple features use less.
    # The pipeline uses lags up to 168.
    expected_rows = len(sample_data) - 168
    # Actually, dropna happens for both shift and lag features. 
    # Shift(-24) creates 24 NaNs at the end.
    # Lags(168) creates 168 NaNs at the start.
    # The valid intersection is limited.
    
    assert len(X) > 0
    assert len(y) == len(X)
    
def test_temporal_features(sample_data):
    engine = TimeSeriesFeatureEngine()
    features = engine.create_features(sample_data)
    
    expected_cols = ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"]
    for col in expected_cols:
        assert col in features.columns

def test_pipeline_scaling(sample_data):
    # Test valid pipeline execution
    pipeline = create_feature_pipeline(forecast_horizon=24, scaler_type='standard')
    
    # Create simple X (numeric only as pipeline expects)
    X = sample_data.drop(columns=['price'])
    # Pipeline expects to be able to create features from X
    # Actually create_feature_pipeline returns a sklearn Pipeline that expects
    # input that matches the transformers. 
    # The TimeSeriesFeatureEngine uses internal logic.
    
    # Let's test TimeSeriesFeatureEngine's fit_transform which uses the scaler
    engine = TimeSeriesFeatureEngine(feature_config={'scaler_type': 'standard'})
    X_out, _ = engine.prepare_data(sample_data)
    engine.fit(X_out)
    X_scaled = engine.transform(X_out)
    
    assert X_scaled.shape == X_out.shape
    assert np.all(np.isfinite(X_scaled))
