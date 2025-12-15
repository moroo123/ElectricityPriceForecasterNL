"""Market-specific feature transformers for electricity price forecasting."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List


class MarketFeatures(BaseEstimator, TransformerMixin):
    """
    Create electricity market-specific features.

    This transformer creates derived features specific to electricity markets:
    - Residual load (load - renewable generation)
    - Renewable penetration ratio
    - Total renewable generation
    - Interaction features

    Parameters
    ----------
    create_residual_load : bool, default=True
        Create residual load features
    create_renewable_ratio : bool, default=True
        Create renewable penetration ratio
    create_interactions : bool, default=True
        Create interaction features
    load_col : str, default='load_forecast'
        Name of load column
    wind_onshore_col : str, default='wind_onshore'
        Name of wind onshore column
    wind_offshore_col : str, default='wind_offshore'
        Name of wind offshore column
    solar_col : str, default='solar'
        Name of solar column
    """

    def __init__(
        self,
        create_residual_load: bool = True,
        create_renewable_ratio: bool = True,
        create_interactions: bool = True,
        load_col: str = 'load_forecast',
        wind_onshore_col: str = 'wind_onshore',
        wind_offshore_col: str = 'wind_offshore',
        solar_col: str = 'solar'
    ):
        self.create_residual_load = create_residual_load
        self.create_renewable_ratio = create_renewable_ratio
        self.create_interactions = create_interactions
        self.load_col = load_col
        self.wind_onshore_col = wind_onshore_col
        self.wind_offshore_col = wind_offshore_col
        self.solar_col = solar_col

    def fit(self, X, y=None):
        """Fit the transformer (no-op, returns self)."""
        return self

    def transform(self, X):
        """
        Create market-specific features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features with load and renewable generation columns

        Returns
        -------
        pd.DataFrame
            DataFrame with market features
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        result = pd.DataFrame(index=X.index)

        # Calculate total renewable generation
        renewable_cols = []
        if self.wind_onshore_col in X.columns:
            renewable_cols.append(self.wind_onshore_col)
        if self.wind_offshore_col in X.columns:
            renewable_cols.append(self.wind_offshore_col)
        if self.solar_col in X.columns:
            renewable_cols.append(self.solar_col)

        if renewable_cols:
            total_renewable = X[renewable_cols].sum(axis=1)
            result['total_renewable'] = total_renewable

            # Total wind
            wind_cols = [col for col in [self.wind_onshore_col, self.wind_offshore_col]
                        if col in X.columns]
            if wind_cols:
                result['total_wind'] = X[wind_cols].sum(axis=1)

        # Residual load (load - renewables)
        if self.create_residual_load and self.load_col in X.columns and renewable_cols:
            result['residual_load'] = X[self.load_col] - total_renewable

            # Residual load as percentage of total load
            result['residual_load_pct'] = (result['residual_load'] / X[self.load_col]) * 100

        # Renewable penetration ratio
        if self.create_renewable_ratio and self.load_col in X.columns and renewable_cols:
            result['renewable_penetration'] = (total_renewable / X[self.load_col]) * 100

            # Individual renewable ratios
            if self.wind_onshore_col in X.columns:
                result['wind_onshore_ratio'] = (X[self.wind_onshore_col] / X[self.load_col]) * 100
            if self.wind_offshore_col in X.columns:
                result['wind_offshore_ratio'] = (X[self.wind_offshore_col] / X[self.load_col]) * 100
            if self.solar_col in X.columns:
                result['solar_ratio'] = (X[self.solar_col] / X[self.load_col]) * 100

        # Interaction features with hour (if index is datetime)
        if self.create_interactions and isinstance(X.index, pd.DatetimeIndex):
            hour = X.index.hour

            if 'residual_load' in result.columns:
                result['residual_load_x_hour'] = result['residual_load'] * hour

            if self.solar_col in X.columns:
                # Solar × hour captures diurnal pattern
                result['solar_x_hour'] = X[self.solar_col] * hour

            if 'total_wind' in result.columns:
                # Wind × hour
                result['wind_x_hour'] = result['total_wind'] * hour

        # Replace inf values with NaN (can happen with division)
        result = result.replace([np.inf, -np.inf], np.nan)

        return result


class CapacityFactorFeatures(BaseEstimator, TransformerMixin):
    """
    Create capacity factor features for renewable generation.

    Capacity factors indicate how much of the theoretical maximum
    generation is being produced.

    Parameters
    ----------
    capacity_mw : dict
        Dictionary mapping column names to installed capacity in MW
        Example: {'wind_onshore': 6000, 'wind_offshore': 3000, 'solar': 8000}
    """

    def __init__(self, capacity_mw: dict):
        self.capacity_mw = capacity_mw

    def fit(self, X, y=None):
        """Fit the transformer (no-op, returns self)."""
        return self

    def transform(self, X):
        """
        Create capacity factor features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features with renewable generation columns

        Returns
        -------
        pd.DataFrame
            DataFrame with capacity factor features
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        result = pd.DataFrame(index=X.index)

        for col, capacity in self.capacity_mw.items():
            if col in X.columns:
                result[f"{col}_capacity_factor"] = (X[col] / capacity) * 100

        return result


class PriceSpreadFeatures(BaseEstimator, TransformerMixin):
    """
    Create price spread and volatility features.

    These features capture market dynamics and volatility.

    Parameters
    ----------
    price_col : str, default='price'
        Name of the price column
    windows : list of int, default=[24, 168]
        Windows for computing spreads and volatility
    """

    def __init__(
        self,
        price_col: str = 'price',
        windows: List[int] = None
    ):
        self.price_col = price_col
        self.windows = windows or [24, 168]

    def fit(self, X, y=None):
        """Fit the transformer (no-op, returns self)."""
        return self

    def transform(self, X):
        """
        Create price spread features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features with price column

        Returns
        -------
        pd.DataFrame
            DataFrame with price spread features
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        if self.price_col not in X.columns:
            raise ValueError(f"Price column '{self.price_col}' not found in X")

        result = pd.DataFrame(index=X.index)
        price = X[self.price_col]

        for window in self.windows:
            # Price spread from rolling mean
            rolling_mean = price.rolling(window=window, min_periods=1).mean()
            result[f'price_spread_{window}h'] = price - rolling_mean

            # Price spread from rolling min/max
            rolling_min = price.rolling(window=window, min_periods=1).min()
            rolling_max = price.rolling(window=window, min_periods=1).max()

            result[f'price_range_{window}h'] = rolling_max - rolling_min
            result[f'price_position_{window}h'] = (price - rolling_min) / (rolling_max - rolling_min + 1e-6)

            # Coefficient of variation (normalized volatility)
            rolling_std = price.rolling(window=window, min_periods=1).std()
            result[f'price_cv_{window}h'] = rolling_std / (rolling_mean + 1e-6)

        # Replace inf values
        result = result.replace([np.inf, -np.inf], np.nan)

        return result


class LoadForecastErrorFeatures(BaseEstimator, TransformerMixin):
    """
    Create features based on load forecast errors.

    Parameters
    ----------
    forecast_col : str, default='load_forecast'
        Name of load forecast column
    actual_col : str, default='actual_load'
        Name of actual load column
    windows : list of int, default=[24, 168]
        Windows for computing error statistics
    """

    def __init__(
        self,
        forecast_col: str = 'load_forecast',
        actual_col: str = 'actual_load',
        windows: List[int] = None
    ):
        self.forecast_col = forecast_col
        self.actual_col = actual_col
        self.windows = windows or [24, 168]

    def fit(self, X, y=None):
        """Fit the transformer (no-op, returns self)."""
        return self

    def transform(self, X):
        """
        Create load forecast error features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features with forecast and actual load

        Returns
        -------
        pd.DataFrame
            DataFrame with forecast error features
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        result = pd.DataFrame(index=X.index)

        if self.forecast_col in X.columns and self.actual_col in X.columns:
            # Absolute error
            error = X[self.forecast_col] - X[self.actual_col]
            result['load_forecast_error'] = error

            # Recent error statistics
            for window in self.windows:
                result[f'load_mae_{window}h'] = error.abs().rolling(
                    window=window, min_periods=1
                ).mean()

                result[f'load_bias_{window}h'] = error.rolling(
                    window=window, min_periods=1
                ).mean()

        return result
