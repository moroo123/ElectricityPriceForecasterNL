import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def mock_config(monkeypatch):
    """Mock configuration values."""
    monkeypatch.setenv("ENTSOE_API_KEY", "mock_key")
    monkeypatch.setenv("BIDDING_ZONE", "NL")
    
@pytest.fixture
def sample_hourly_data():
    """Create sample data with hourly frequency."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "price": np.random.uniform(0, 100, 100),
        "load_forecast": np.random.uniform(10000, 20000, 100),
        "actual_load": np.random.uniform(10000, 20000, 100),
        "wind_onshore": np.random.uniform(0, 5000, 100),
        "solar": np.random.uniform(0, 5000, 100)
    }, index=dates)
    return df
