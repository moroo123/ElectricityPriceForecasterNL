import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.preprocessing import split_train_val_test, handle_missing_values

def test_split_train_val_test_dates():
    dates = pd.date_range(start="2023-01-01", end="2024-01-31", freq="h")
    df = pd.DataFrame({"value": range(len(dates))}, index=dates)
    
    train, val, test = split_train_val_test(
        df,
        train_end="2023-06-30",
        val_end="2023-09-30"
    )
    
    assert train.index.max() <= pd.Timestamp("2023-06-30")
    assert val.index.min() > pd.Timestamp("2023-06-30")
    assert val.index.max() <= pd.Timestamp("2023-09-30")
    assert test.index.min() > pd.Timestamp("2023-09-30")
    
    # Check no leakage/overlap
    assert len(set(train.index) & set(val.index)) == 0
    assert len(set(val.index) & set(test.index)) == 0

def test_handle_missing_values():
    df = pd.DataFrame({
        "a": [1, np.nan, 3, np.nan, 5],
        "b": [1, 2, 3, 4, 5]
    })
    
    # Forward fill
    df_ffill = handle_missing_values(df, strategy="forward_fill")
    assert df_ffill["a"].isna().sum() == 0
    assert df_ffill.iloc[1]["a"] == 1
    
    # Zero fill
    df_zero = handle_missing_values(df, strategy="zero")
    assert df_zero.iloc[1]["a"] == 0
