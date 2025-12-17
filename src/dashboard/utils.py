import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.entsoe_client import EntsoeDataClient
from src.data.cache import DataCache
from src.config import Config


@st.cache_resource
def load_model_cached(model_path: Path):
    """Load model and feature engine with caching."""
    import xgboost as xgb
    
    # Load model
    model = xgb.XGBRegressor()
    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "regressor"
    model.load_model(str(model_path))
    
    # Load feature engine
    fe_path = model_path.parent / f"{model_path.stem}_feature_engine.pkl"
    with open(fe_path, 'rb') as f:
        feature_engine = pickle.load(f)
        
    return model, feature_engine


def update_recent_data(days_ahead: int = 3):
    """
    Fetch missing recent data and update the database.
    
    Parameters
    ----------
    days_ahead : int, default=3
        Number of days ahead to fetch (for forecasts)
    """
    client = EntsoeDataClient(bidding_zone=Config.BIDDING_ZONE)
    cache = DataCache()
    
    # Define mapping of data types to client methods
    data_types = {
        "prices": client.get_day_ahead_prices,
        "load_forecast": client.get_load_forecast,
        "actual_load": client.get_actual_load,
        "wind_onshore": client.get_wind_forecast,
        "wind_offshore": client.get_wind_offshore_forecast,
        "solar": client.get_solar_forecast,
    }
    
    now = pd.Timestamp.now(tz="UTC")
    end_date = now + pd.Timedelta(days=days_ahead)
    
    status_msg = []
    
    for data_type, fetch_func in data_types.items():
        try:
            # Determine start date based on last available data
            last_ts = cache.get_last_timestamp(data_type, Config.BIDDING_ZONE)
            
            if last_ts is None:
                # Default start if no data exists (e.g., start of this year)
                start_date = pd.Timestamp(f"{now.year}-01-01", tz="UTC")
            else:
                # Start from the last timestamp (inclusive) to ensure no gaps.
                # The database handles duplicates via UPSERT.
                start_date = last_ts
            
            # Only fetch if there's a gap or we need future data
            if start_date < end_date:
                # Avoid fetching if the gap is negligible (less than 1 hour)
                if (end_date - start_date) < pd.Timedelta(hours=1):
                    continue
                    
                # print(f"Updating {data_type}: {start_date} -> {end_date}")
                data = fetch_func(start_date, end_date)
                
                if not data.empty:
                    cache.store_data(data_type, data, Config.BIDDING_ZONE)
                    status_msg.append(f"Updated {data_type} ({len(data)} records)")
                    
        except Exception as e:
            # Log error but continue with other data types
            print(f"Error updating {data_type}: {e}")
            status_msg.append(f"Failed to update {data_type}: {str(e)}")
            
    return status_msg


def get_live_data(hours_history: int = 48, hours_forecast: int = 24):
    """
    Get recent data from database for live forecasting.
    
    Returns
    -------
    pd.DataFrame
        Merged dataframe with history and forecast data
    """
    from src.utils.preprocessing import create_merged_dataset, handle_missing_values
    
    # We can reuse the existing create_merged_dataset function
    # It fetches everything from DB. We just need to filter it.
    df = create_merged_dataset(db_path=Config.DB_PATH)
    
    # Handle missing values (critical for live inference)
    # Using the same strategy as training: forward fill
    df = handle_missing_values(df, strategy='forward_fill', limit=24)
    
    # Filter for relevant window
    now = pd.Timestamp.now(tz="Europe/Amsterdam")
    start_time = now - pd.Timedelta(hours=hours_history)
    end_time = now + pd.Timedelta(hours=hours_forecast)
    
    # Ensure index is localized
    if df.index.tz is None:
        df.index = df.index.tz_localize("Europe/Amsterdam")
    else:
        df.index = df.index.tz_convert("Europe/Amsterdam")
        
    mask = (df.index >= start_time) & (df.index <= end_time)
    return df[mask]
