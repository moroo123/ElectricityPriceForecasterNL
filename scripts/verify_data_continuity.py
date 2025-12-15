
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from utils.preprocessing import create_merged_dataset, handle_missing_values

def check_continuity():
    print("Checking data continuity...")
    df = create_merged_dataset(db_path="data/cache.db")
    
    print(f"Original shape: {df.shape}")
    print(f"Index: {df.index.min()} to {df.index.max()}")
    
    # Check frequency
    expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    print(f"Expected hourly rows: {len(expected_range)}")
    print(f"Actual rows: {len(df)}")
    
    if len(df) != len(expected_range):
        print(f"MISSING ROWS DETECTED: {len(expected_range) - len(df)}")
        missing_idx = expected_range.difference(df.index)
        print(f"First 5 missing timestamps: {missing_idx[:5]}")
    else:
        print("Data is continuous (no missing rows in original).")

    # Simulate cleaning process in notebook
    print("\nSimulating notebook cleaning...")
    df_clean = handle_missing_values(df, strategy='forward_fill', limit=24)
    df_dropped = df_clean.dropna()
    
    print(f"Shape after dropna(): {df_dropped.shape}")
    
    if len(df_dropped) != len(expected_range):
        diff = len(expected_range) - len(df_dropped)
        print(f"WARNING: dropna() removed {diff} rows!")
        print("This breaks integer-based shifting (lags)!")
        
        # Verify lag misalignment
        # If we shift by 168 rows, is it 168 hours?
        # Let's take a sample point
        # index i
        # index i-168
        # diff should be 168 hours
        
        delta = df_dropped.index[168] - df_dropped.index[0]
        print(f"Time difference for 168-row shift at start: {delta}")
        if delta != pd.Timedelta(hours=168):
            print("CRITICAL: 168-row shift IS NOT 168 hours!")
    
if __name__ == "__main__":
    check_continuity()
