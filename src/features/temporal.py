"""Temporal feature transformers for time-series data."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional


class TemporalFeatures(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from datetime index.

    Creates features like hour, day of week, month, weekend flag, etc.
    Uses cyclical encoding (sin/cos) for periodic features to maintain continuity.

    Parameters
    ----------
    features : list of str, optional
        List of features to create. Options:
        - 'hour': Hour of day (0-23)
        - 'day_of_week': Day of week (0-6, Monday=0)
        - 'month': Month (1-12)
        - 'day_of_year': Day of year (1-365/366)
        - 'weekend': Weekend flag (0/1)
        - 'quarter': Quarter (1-4)
        If None, creates all features.
    cyclical : bool, default=True
        Whether to use cyclical encoding (sin/cos) for periodic features.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        cyclical: bool = True
    ):
        self.features = features
        self.cyclical = cyclical

    def fit(self, X, y=None):
        """Fit the transformer (no-op, returns self)."""
        return self

    def transform(self, X):
        """
        Transform datetime index into temporal features.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with DatetimeIndex

        Returns
        -------
        pd.DataFrame
            DataFrame with temporal features
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")

        features_to_create = self.features or [
            'hour', 'day_of_week', 'month', 'day_of_year', 'weekend'
        ]

        result = pd.DataFrame(index=X.index)

        for feature in features_to_create:
            if feature == 'hour':
                if self.cyclical:
                    result['hour_sin'] = np.sin(2 * np.pi * X.index.hour / 24)
                    result['hour_cos'] = np.cos(2 * np.pi * X.index.hour / 24)
                else:
                    result['hour'] = X.index.hour

            elif feature == 'day_of_week':
                if self.cyclical:
                    result['day_of_week_sin'] = np.sin(2 * np.pi * X.index.dayofweek / 7)
                    result['day_of_week_cos'] = np.cos(2 * np.pi * X.index.dayofweek / 7)
                else:
                    result['day_of_week'] = X.index.dayofweek

            elif feature == 'month':
                if self.cyclical:
                    result['month_sin'] = np.sin(2 * np.pi * X.index.month / 12)
                    result['month_cos'] = np.cos(2 * np.pi * X.index.month / 12)
                else:
                    result['month'] = X.index.month

            elif feature == 'day_of_year':
                if self.cyclical:
                    # Account for leap years
                    days_in_year = 365 + X.index.is_leap_year.astype(int)
                    result['day_of_year_sin'] = np.sin(2 * np.pi * X.index.dayofyear / days_in_year)
                    result['day_of_year_cos'] = np.cos(2 * np.pi * X.index.dayofyear / days_in_year)
                else:
                    result['day_of_year'] = X.index.dayofyear

            elif feature == 'weekend':
                result['is_weekend'] = (X.index.dayofweek >= 5).astype(int)

            elif feature == 'quarter':
                if self.cyclical:
                    result['quarter_sin'] = np.sin(2 * np.pi * X.index.quarter / 4)
                    result['quarter_cos'] = np.cos(2 * np.pi * X.index.quarter / 4)
                else:
                    result['quarter'] = X.index.quarter

        return result

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        # This will be determined during transform
        return None


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode cyclical features using sin/cos transformation.

    Parameters
    ----------
    period : float
        The period of the cyclical feature (e.g., 24 for hours, 7 for days)
    """

    def __init__(self, period: float):
        self.period = period

    def fit(self, X, y=None):
        """Fit the encoder (no-op, returns self)."""
        return self

    def transform(self, X):
        """
        Transform features using sin/cos encoding.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Cyclical feature to encode

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Sin and cos encoded features
        """
        X = np.asarray(X).reshape(-1, 1)
        sin_encoded = np.sin(2 * np.pi * X / self.period)
        cos_encoded = np.cos(2 * np.pi * X / self.period)
        return np.hstack([sin_encoded, cos_encoded])


class HolidayFeatures(BaseEstimator, TransformerMixin):
    """
    Create features for Dutch public holidays.

    Parameters
    ----------
    country : str, default='NL'
        Country code for holidays
    """

    def __init__(self, country: str = 'NL'):
        self.country = country
        self._holidays = None

    def fit(self, X, y=None):
        """Fit the transformer and load holiday data."""
        try:
            import holidays
            self._holidays = holidays.country_holidays(self.country)
        except ImportError:
            # If holidays package not available, skip this feature
            self._holidays = None
        return self

    def transform(self, X):
        """
        Create holiday features.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with DatetimeIndex

        Returns
        -------
        pd.DataFrame
            DataFrame with holiday features
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")

        result = pd.DataFrame(index=X.index)

        if self._holidays is not None:
            # Use pd.Index to leverage .isin method
            result['is_holiday'] = pd.Index(X.index.date).isin(self._holidays).astype(int)
            
            # Day before holiday (shift index forward by 1 day)
            result['is_pre_holiday'] = pd.Index((X.index + pd.Timedelta(days=1)).date).isin(self._holidays).astype(int)
            
            # Day after holiday (shift index backward by 1 day)
            result['is_post_holiday'] = pd.Index((X.index - pd.Timedelta(days=1)).date).isin(self._holidays).astype(int)
        else:
            result['is_holiday'] = 0
            result['is_pre_holiday'] = 0
            result['is_post_holiday'] = 0

        return result
