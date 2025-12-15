"""Lag and rolling window feature transformers."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Union


class LagFeatures(BaseEstimator, TransformerMixin):
    """
    Create lag features from time-series data.

    IMPORTANT: This transformer is designed to prevent data leakage.
    It should only be used on the target variable or features that are
    available at prediction time with appropriate lag periods.

    Parameters
    ----------
    lags : list of int
        List of lag periods to create (e.g., [1, 2, 24, 48])
    columns : list of str, optional
        Columns to create lags for. If None, uses all columns.
    fill_value : float, default=np.nan
        Value to fill for initial NaN values created by lagging
    """

    def __init__(
        self,
        lags: List[int],
        columns: Optional[List[str]] = None,
        fill_value: float = np.nan
    ):
        self.lags = lags
        self.columns = columns
        self.fill_value = fill_value

    def fit(self, X, y=None):
        """Fit the transformer (no-op, returns self)."""
        return self

    def transform(self, X):
        """
        Create lag features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        pd.DataFrame
            DataFrame with lag features
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        columns_to_lag = self.columns or X.columns.tolist()
        result = pd.DataFrame(index=X.index)

        for col in columns_to_lag:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in X")

            for lag in self.lags:
                lag_col_name = f"{col}_lag_{lag}"
                result[lag_col_name] = X[col].shift(lag).fillna(self.fill_value)

        return result


class RollingFeatures(BaseEstimator, TransformerMixin):
    """
    Create rolling window features from time-series data.

    Parameters
    ----------
    windows : list of int
        Window sizes for rolling computations (e.g., [24, 48, 168])
    functions : list of str, default=['mean', 'std', 'min', 'max']
        Aggregation functions to apply. Options: 'mean', 'std', 'min', 'max', 'median'
    columns : list of str, optional
        Columns to create rolling features for. If None, uses all columns.
    min_periods : int, optional
        Minimum number of observations required. If None, equals window size.
    fill_value : float, default=np.nan
        Value to fill for initial NaN values
    """

    def __init__(
        self,
        windows: List[int],
        functions: List[str] = None,
        columns: Optional[List[str]] = None,
        min_periods: Optional[int] = None,
        fill_value: float = np.nan
    ):
        self.windows = windows
        self.functions = functions or ['mean', 'std', 'min', 'max']
        self.columns = columns
        self.min_periods = min_periods
        self.fill_value = fill_value

    def fit(self, X, y=None):
        """Fit the transformer (no-op, returns self)."""
        return self

    def transform(self, X):
        """
        Create rolling window features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        pd.DataFrame
            DataFrame with rolling features
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        columns_to_roll = self.columns or X.columns.tolist()
        result = pd.DataFrame(index=X.index)

        for col in columns_to_roll:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in X")

            for window in self.windows:
                min_periods = self.min_periods or window

                for func in self.functions:
                    feature_name = f"{col}_rolling_{window}_{func}"

                    if func == 'mean':
                        result[feature_name] = X[col].rolling(
                            window=window, min_periods=min_periods
                        ).mean().fillna(self.fill_value)

                    elif func == 'std':
                        result[feature_name] = X[col].rolling(
                            window=window, min_periods=min_periods
                        ).std().fillna(self.fill_value)

                    elif func == 'min':
                        result[feature_name] = X[col].rolling(
                            window=window, min_periods=min_periods
                        ).min().fillna(self.fill_value)

                    elif func == 'max':
                        result[feature_name] = X[col].rolling(
                            window=window, min_periods=min_periods
                        ).max().fillna(self.fill_value)

                    elif func == 'median':
                        result[feature_name] = X[col].rolling(
                            window=window, min_periods=min_periods
                        ).median().fillna(self.fill_value)

                    else:
                        raise ValueError(f"Unknown function: {func}")

        return result


class DifferenceFeatures(BaseEstimator, TransformerMixin):
    """
    Create difference features for time-series data.

    Parameters
    ----------
    periods : list of int
        Periods for differencing (e.g., [1, 24] for 1-hour and 24-hour differences)
    columns : list of str, optional
        Columns to create differences for. If None, uses all columns.
    fill_value : float, default=np.nan
        Value to fill for initial NaN values
    """

    def __init__(
        self,
        periods: List[int],
        columns: Optional[List[str]] = None,
        fill_value: float = np.nan
    ):
        self.periods = periods
        self.columns = columns
        self.fill_value = fill_value

    def fit(self, X, y=None):
        """Fit the transformer (no-op, returns self)."""
        return self

    def transform(self, X):
        """
        Create difference features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        pd.DataFrame
            DataFrame with difference features
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        columns_to_diff = self.columns or X.columns.tolist()
        result = pd.DataFrame(index=X.index)

        for col in columns_to_diff:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in X")

            for period in self.periods:
                diff_col_name = f"{col}_diff_{period}"
                result[diff_col_name] = X[col].diff(period).fillna(self.fill_value)

        return result


class ExpandingFeatures(BaseEstimator, TransformerMixin):
    """
    Create expanding window features (cumulative statistics).

    Parameters
    ----------
    functions : list of str, default=['mean', 'std']
        Aggregation functions to apply
    columns : list of str, optional
        Columns to create expanding features for. If None, uses all columns.
    min_periods : int, default=1
        Minimum number of observations required
    """

    def __init__(
        self,
        functions: List[str] = None,
        columns: Optional[List[str]] = None,
        min_periods: int = 1
    ):
        self.functions = functions or ['mean', 'std']
        self.columns = columns
        self.min_periods = min_periods

    def fit(self, X, y=None):
        """Fit the transformer (no-op, returns self)."""
        return self

    def transform(self, X):
        """
        Create expanding window features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        pd.DataFrame
            DataFrame with expanding features
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        columns_to_expand = self.columns or X.columns.tolist()
        result = pd.DataFrame(index=X.index)

        for col in columns_to_expand:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in X")

            for func in self.functions:
                feature_name = f"{col}_expanding_{func}"

                if func == 'mean':
                    result[feature_name] = X[col].expanding(
                        min_periods=self.min_periods
                    ).mean()
                elif func == 'std':
                    result[feature_name] = X[col].expanding(
                        min_periods=self.min_periods
                    ).std()
                elif func == 'min':
                    result[feature_name] = X[col].expanding(
                        min_periods=self.min_periods
                    ).min()
                elif func == 'max':
                    result[feature_name] = X[col].expanding(
                        min_periods=self.min_periods
                    ).max()

        return result
