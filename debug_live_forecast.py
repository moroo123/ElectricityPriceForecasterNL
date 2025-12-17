
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import streamlit as st

# Setup path
sys.path.append(str(Path.cwd()))

from src.dashboard.utils import get_live_data, update_recent_data
from src.config import Config
from src.models.train import predict_future

def load_model_debug():
    # Find a model
    model_dir = Path("models")
    model_files = list(model_dir.glob("*.json"))
    if not model_files:
        print("No models found!")
        return None, None
        
    model_path = model_files[0] # Pick the first one
    print(f"Loading model: {model_path}")
    
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    
    fe_path = model_path.parent / f"{model_path.stem}_feature_engine.pkl"
    with open(fe_path, 'rb') as f:
        feature_engine = pickle.load(f)
        
    return model, feature_engine

def main():
    print("--- Debugging Live Forecast ---")
    
    # 1. Update Data
    print("Updating data...")
    update_recent_data(days_ahead=3)
    
    # 2. Get Live Data
    print("Fetching live data (336h history)...")
    hours_history = 336
    df = get_live_data(hours_history=hours_history, hours_forecast=48)
    
    print(f"Data shape: {df.shape}")
    print(f"Index range: {df.index.min()} to {df.index.max()}")
    print("Missing values in raw data:")
    print(df.isna().sum())
    
    # 3. Load Model
    model, fe = load_model_debug()
    if not model:
        return

    # 4. Generate Features
    print("\nGenerating features...")
    try:
        # We manually call prepare_data to inspect features
        X, _ = fe.prepare_data(df, create_target=False)
        print(f"Features shape: {X.shape}")
        
        # Check for NaNs
        nan_counts = X.isna().sum()
        with_nans = nan_counts[nan_counts > 0]
        if not with_nans.empty:
            print("\n⚠️ Features with NaNs:")
            print(with_nans)
        else:
            print("\n✅ No NaNs in features.")
            
        # Check for weird values
        print("\nFeature Statistics:")
        print(X.describe().T[['mean', 'min', 'max']])
        
        # 5. Predict
        print("\nPredicting...")
        preds = model.predict(X)
        pred_series = pd.Series(preds, index=X.index + pd.Timedelta(hours=24)) # Assuming 24h horizon
        
        print("\nPredictions (Next 24h):")
        now = pd.Timestamp.now(tz="Europe/Amsterdam")
        future_preds = pred_series[pred_series.index > now].head(24)
        print(future_preds)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
