"""Data preprocessing utilities for electricity price forecasting."""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from typing import Tuple, Optional, Dict


def load_data_from_cache(
    db_path: str = "data/cache.db",
    table_name: str = "prices"
) -> pd.DataFrame:
    """
    Load data from SQLite cache.

    Parameters
    ----------
    db_path : str
        Path to SQLite database
    table_name : str
        Name of table to load

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex in Europe/Amsterdam timezone
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df.index = df.index.tz_convert("Europe/Amsterdam")

    return df


def resample_to_hourly(
    df: pd.DataFrame,
    value_col: str = "value",
    agg_func: str = "mean"
) -> pd.Series:
    """
    Resample time-series data to hourly frequency.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    value_col : str
        Column to resample
    agg_func : str
        Aggregation function ('mean', 'sum', 'first', 'last')

    Returns
    -------
    pd.Series
        Resampled hourly series
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")

    if agg_func == "mean":
        return df[value_col].resample("h").mean()
    elif agg_func == "sum":
        return df[value_col].resample("h").sum()
    elif agg_func == "first":
        return df[value_col].resample("h").first()
    elif agg_func == "last":
        return df[value_col].resample("h").last()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")


def create_merged_dataset(
    db_path: str = "data/cache.db",
    resample: bool = True
) -> pd.DataFrame:
    """
    Create merged dataset from all tables in cache database.

    Parameters
    ----------
    db_path : str
        Path to SQLite database
    resample : bool
        Whether to resample all data to hourly

    Returns
    -------
    pd.DataFrame
        Merged dataset with all features
    """
    # Table names and their corresponding column names in merged dataset
    tables = {
        "prices": "price",
        "load_forecast": "load_forecast",
        "actual_load": "actual_load",
        "wind_onshore": "wind_onshore",
        "wind_offshore": "wind_offshore",
        "solar": "solar",
    }

    series_list = []

    for table_name, col_name in tables.items():
        try:
            df = load_data_from_cache(db_path, table_name)

            if not df.empty:
                if resample:
                    series = resample_to_hourly(df, "value", "mean")
                else:
                    series = df["value"]

                series.name = col_name
                series_list.append(series)
        except Exception as e:
            print(f"Warning: Could not load {table_name}: {e}")

    if not series_list:
        raise ValueError("No data could be loaded from database")

    # Merge all series
    merged = pd.DataFrame(series_list).T

    # Drop rows with any NaN values (or you could use forward fill)
    # For now, we'll keep NaNs and let the feature pipeline handle them

    return merged


def split_train_val_test(
    df: pd.DataFrame,
    train_end: str = "2023-12-31",
    val_end: str = "2024-06-30",
    test_end: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data into train, validation, and test sets.

    IMPORTANT: This performs a time-based split without shuffling
    to prevent data leakage in time-series forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    train_end : str
        End date for training set (inclusive)
    val_end : str
        End date for validation set (inclusive)
    test_end : str, optional
        End date for test set (inclusive). If None, uses all remaining data.

    Returns
    -------
    train : pd.DataFrame
        Training set
    val : pd.DataFrame
        Validation set
    test : pd.DataFrame
        Test set

    Example
    -------
    >>> train, val, test = split_train_val_test(
    ...     df,
    ...     train_end="2023-12-31",
    ...     val_end="2024-06-30"
    ... )
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Convert string dates to datetime
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)

    # Ensure dates are timezone-aware if df.index is timezone-aware
    if df.index.tz is not None:
        train_end_dt = train_end_dt.tz_localize(df.index.tz)
        val_end_dt = val_end_dt.tz_localize(df.index.tz)
        if test_end is not None:
            test_end_dt = pd.to_datetime(test_end).tz_localize(df.index.tz)

    # Split data
    train = df[df.index <= train_end_dt]
    val = df[(df.index > train_end_dt) & (df.index <= val_end_dt)]

    if test_end is not None:
        test = df[(df.index > val_end_dt) & (df.index <= test_end_dt)]
    else:
        test = df[df.index > val_end_dt]

    print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} samples)")
    print(f"Val:   {val.index.min()} to {val.index.max()} ({len(val)} samples)")
    print(f"Test:  {test.index.min()} to {test.index.max()} ({len(test)} samples)")

    return train, val, test


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "forward_fill",
    limit: Optional[int] = 24
) -> pd.DataFrame:
    """
    Handle missing values in time-series data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential missing values
    strategy : str
        Strategy for handling missing values:
        - 'forward_fill': Forward fill with limit
        - 'interpolate': Linear interpolation
        - 'drop': Drop rows with any NaN
        - 'zero': Fill with zeros
    limit : int, optional
        Maximum number of consecutive NaNs to fill

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled
    """
    df_clean = df.copy()

    if strategy == "forward_fill":
        df_clean = df_clean.ffill(limit=limit)
    elif strategy == "interpolate":
        df_clean = df_clean.interpolate(method="linear", limit=limit)
    elif strategy == "drop":
        df_clean = df_clean.dropna()
    elif strategy == "zero":
        df_clean = df_clean.fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df_clean


def detect_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers in a time series.

    Parameters
    ----------
    series : pd.Series
        Time series data
    method : str
        Method for outlier detection:
        - 'iqr': Interquartile range method
        - 'zscore': Z-score method
        - 'rolling': Rolling statistics method
    threshold : float
        Threshold for outlier detection

    Returns
    -------
    pd.Series
        Boolean series indicating outliers (True = outlier)
    """
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)

    elif method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold

    elif method == "rolling":
        # Detect values that are far from rolling mean
        rolling_mean = series.rolling(window=24, min_periods=1).mean()
        rolling_std = series.rolling(window=24, min_periods=1).std()
        z_scores = np.abs((series - rolling_mean) / (rolling_std + 1e-6))
        return z_scores > threshold

    else:
        raise ValueError(f"Unknown method: {method}")


def create_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 168  # 1 week
) -> list:
    """
    Create time-series cross-validation splits.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    n_splits : int
        Number of CV splits
    test_size : int
        Size of test set in each split (in hours)

    Returns
    -------
    list
        List of (train_idx, test_idx) tuples
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    splits = []
    total_size = len(df)
    step_size = (total_size - test_size) // n_splits

    for i in range(n_splits):
        test_start = step_size * (i + 1)
        test_end = test_start + test_size

        if test_end > total_size:
            break

        train_idx = df.index[:test_start]
        test_idx = df.index[test_start:test_end]

        splits.append((train_idx, test_idx))

    return splits


def add_lagged_target(
    df: pd.DataFrame,
    target_col: str,
    horizon: int = 24
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Add target variable with specified forecast horizon.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    target_col : str
        Name of target column
    horizon : int
        Forecast horizon in hours

    Returns
    -------
    X : pd.DataFrame
        Features without target
    y : pd.Series
        Target variable shifted by horizon
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    # Create target (future values)
    y = df[target_col].shift(-horizon)

    # Remove rows where target is NaN
    valid_idx = ~y.isna()
    X = df[valid_idx].drop(columns=[target_col])
    y = y[valid_idx]

    return X, y
