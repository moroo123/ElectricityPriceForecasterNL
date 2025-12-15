"""Main feature pipeline for electricity price forecasting."""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List, Dict

from .temporal import TemporalFeatures, HolidayFeatures
from .lag_features import LagFeatures, RollingFeatures
from .market_features import MarketFeatures, LoadForecastErrorFeatures


class DataFrameFeatureUnion:
    """
    Combines multiple DataFrame transformers horizontally.
    Similar to FeatureUnion but preserves DataFrame structure.
    """

    def __init__(self, transformer_list):
        """
        Parameters
        ----------
        transformer_list : list of tuples
            List of (name, transformer) tuples
        """
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        """Fit all transformers."""
        for name, transformer in self.transformer_list:
            transformer.fit(X, y)
        return self

    def transform(self, X):
        """Transform X using all transformers and concatenate results."""
        results = []
        for name, transformer in self.transformer_list:
            result = transformer.transform(X)
            if isinstance(result, pd.DataFrame):
                results.append(result)
            else:
                # Convert to DataFrame if not already
                if hasattr(transformer, 'get_feature_names_out'):
                    cols = transformer.get_feature_names_out()
                else:
                    cols = [f"{name}_{i}" for i in range(result.shape[1])]
                results.append(pd.DataFrame(result, index=X.index, columns=cols))

        return pd.concat(results, axis=1)

    def fit_transform(self, X, y=None):
        """Fit and transform."""
        return self.fit(X, y).transform(X)


def create_feature_pipeline(
    target_col: str = 'price',
    forecast_horizon: int = 24,
    price_lags: Optional[List[int]] = None,
    load_lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    include_holidays: bool = True,
    scaler_type: str = 'robust'
) -> Pipeline:
    """
    Create a complete feature engineering pipeline for electricity price forecasting.

    Parameters
    ----------
    target_col : str, default='price'
        Name of the target column (will be excluded from features)
    forecast_horizon : int, default=24
        Forecast horizon in hours (affects minimum lag)
    price_lags : list of int, optional
        Lag periods for price features. If None, uses [24, 48, 168]
        Note: All lags should be >= forecast_horizon to avoid data leakage
    load_lags : list of int, optional
        Lag periods for load features. If None, uses [1, 24, 168]
    rolling_windows : list of int, optional
        Window sizes for rolling features. If None, uses [24, 48, 168]
    include_holidays : bool, default=True
        Whether to include holiday features
    scaler_type : str, default='robust'
        Type of scaler to use: 'standard', 'robust', or None

    Returns
    -------
    Pipeline
        Complete feature engineering pipeline

    Notes
    -----
    The pipeline creates the following features:
    1. Temporal features (hour, day of week, month, etc.) with cyclical encoding
    2. Holiday features (if enabled)
    3. Market features (residual load, renewable penetration, etc.)
    4. Lag features for prices and loads
    5. Rolling statistics
    6. Scaling (if enabled)

    Example
    -------
    >>> pipeline = create_feature_pipeline(forecast_horizon=24)
    >>> X_transformed = pipeline.fit_transform(X_train)
    """
    # Default lag periods (all >= forecast_horizon to avoid leakage)
    if price_lags is None:
        # Include recent lags (0 = current time) provided target is shifted
        # 144 = 168 - 24 (Weekly persistence relative to forecast horizon)
        price_lags = [0, 1, 2, 3, 6, 12, 23, 24, 48, 144, 168]
    if load_lags is None:
        load_lags = [0, 1, 6, 12, 23, 24, 168]  # 0 is current load/forecast
    if rolling_windows is None:
        rolling_windows = [24, 48, 168]  # 1 day, 2 days, 1 week


    # Build feature transformers
    transformers = []

    # 1. Temporal features
    transformers.append(
        ('temporal', TemporalFeatures(
            features=['hour', 'day_of_week', 'month', 'weekend'],
            cyclical=True
        ))
    )

    # 2. Holiday features
    if include_holidays:
        transformers.append(
            ('holidays', HolidayFeatures(country='NL'))
        )

    # 3. Market features
    transformers.append(
        ('market', MarketFeatures(
            create_residual_load=True,
            create_renewable_ratio=True,
            create_interactions=True
        ))
    )

    # 4. Load Forecast Error features
    transformers.append(
        ('load_error', LoadForecastErrorFeatures(
            forecast_col='load_forecast',
            actual_col='actual_load'
        ))
    )

    # Create the feature union
    feature_union = DataFrameFeatureUnion(transformers)

    # Build the complete pipeline
    steps = [
        ('features', feature_union),
    ]

    # Add lag features (applied after feature creation)
    # Note: This is a simplified approach. For production, you might want
    # to handle this more carefully to ensure no leakage.

    if scaler_type == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaler_type == 'robust':
        steps.append(('scaler', RobustScaler()))

    return Pipeline(steps)


class TimeSeriesFeatureEngine(BaseEstimator, TransformerMixin):
    """
    Complete feature engineering engine for time-series forecasting.

    This class handles the entire feature engineering workflow including:
    - Data preparation
    - Feature creation
    - Lag and rolling features
    - Train/test splitting with no data leakage
    - Scaling

    Parameters
    ----------
    target_col : str
        Name of the target column
    forecast_horizon : int, default=24
        Forecast horizon in hours
    feature_config : dict, optional
        Configuration for feature creation
    """

    _estimator_type = "transformer"

    def __init__(
        self,
        target_col: str = 'price',
        forecast_horizon: int = 24,
        feature_config: Optional[Dict] = None,
        price_lags: Optional[List[int]] = None,
        price_windows: Optional[List[int]] = None,
        load_lags: Optional[List[int]] = None
    ):
        self.target_col = target_col
        self.forecast_horizon = forecast_horizon
        self.feature_config = feature_config or {}
        # Better default lags: recent prices are critical for day-ahead forecasting
        # Lag 0 = Price at time T (known when predicting T+24)
        self.price_lags = price_lags or [0, 1, 2, 23, 24, 47, 48, 167, 168]
        self.price_windows = price_windows or [24, 48, 168]  # Daily, 2-day, weekly patterns

        self.load_lags = load_lags or [24, 168]  # Day-ahead forecasts available
        self.future_covariates = ["load_forecast"]  # Features known in advance
        self.pipeline = None
        self.feature_columns_ = None
        self.scaler_ = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from raw data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data with DatetimeIndex

        Returns
        -------
        pd.DataFrame
            DataFrame with all features
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        # Start with a copy
        result = df.copy()

        # 1. Temporal features
        temporal_transformer = TemporalFeatures(
            features=['hour', 'day_of_week', 'month', 'weekend'],
            cyclical=True
        )
        temporal_features = temporal_transformer.fit_transform(result)
        result = pd.concat([result, temporal_features], axis=1)

        # 2. Market features
        market_transformer = MarketFeatures()
        market_features = market_transformer.fit_transform(result)
        result = pd.concat([result, market_features], axis=1)

        # 3. Future covariates (shift future values to current row)
        if self.future_covariates:
            for col in self.future_covariates:
                if col in result.columns:
                    # Create future feature aligned with target predictions
                    # Value at t becomes value at t+horizon (which is what we want to predict)
                    # We want X[t] to contain the value of feature at t+horizon.
                    # So we shift by -horizon.
                    future_col = f"{col}_future"
                    result[future_col] = result[col].shift(-self.forecast_horizon)
                    
                    # Also update valid indices to include these shifted values
                    # (Though they will be NaN at the end, which is handled later)

        # 3. Lag features for price (only if target is in data)
        if self.target_col in result.columns:
            price_lag_transformer = LagFeatures(
                lags=self.price_lags,
                columns=[self.target_col],
                fill_value=0
            )
            price_lag_features = price_lag_transformer.fit_transform(result)
            result = pd.concat([result, price_lag_features], axis=1)

            # Rolling features on price
            price_rolling_transformer = RollingFeatures(
                windows=self.price_windows,
                functions=['mean', 'std', 'min', 'max'],
                columns=[self.target_col],
                min_periods=1,
                fill_value=0
            )
            price_rolling_features = price_rolling_transformer.fit_transform(result)
            result = pd.concat([result, price_rolling_features], axis=1)

        # 4. Lag features for load and renewables
        feature_cols = []
        if 'load_forecast' in result.columns:
            feature_cols.append('load_forecast')
        if 'residual_load' in result.columns:
            feature_cols.append('residual_load')
        if 'load_forecast_error' in result.columns:
            feature_cols.append('load_forecast_error')

        if feature_cols:
            # Add future versions to lag list if they exist?
            # actually, we might want lags of the FUTURE forecast too?
            # e.g. change in forecast?
            # For now, let's keep simple lags of existing columns.
            
            load_lag_transformer = LagFeatures(
                lags=self.load_lags,
                columns=feature_cols,
                fill_value=0
            )
            load_lag_features = load_lag_transformer.fit_transform(result)
            result = pd.concat([result, load_lag_features], axis=1)

        # 5. Holiday features
        try:
            holiday_transformer = HolidayFeatures(country='NL')
            holiday_features = holiday_transformer.fit_transform(result)
            result = pd.concat([result, holiday_features], axis=1)
        except Exception:
            # Skip if holidays package not available
            pass

        # 6. Basic price-based features
        if self.target_col in df.columns:
            # Price differences (trends)
            result['price_diff_24h'] = df[self.target_col].diff(24)
            result['price_diff_168h'] = df[self.target_col].diff(168)

        # 7. Basic residual load features
        if 'residual_load' in result.columns:
            # Residual load lags (critical for pricing!)
            for lag in [24, 48, 168]:
                result[f'residual_load_lag_{lag}h'] = result['residual_load'].shift(lag)

        if 'renewable_penetration' in result.columns:
            # Renewable penetration categories (helps with non-linear patterns)
            result['high_renewable_penetration'] = (result['renewable_penetration'] > 50).astype(int)
            result['very_high_renewable_penetration'] = (result['renewable_penetration'] > 75).astype(int)

            # Renewable penetration lags
            result['renewable_penetration_lag_24h'] = result['renewable_penetration'].shift(24)


        return result

    def prepare_data(
        self,
        df: pd.DataFrame,
        create_target: bool = True
    ) -> tuple:
        """
        Prepare data for modeling by creating features and target.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data
        create_target : bool, default=True
            Whether to create the target variable

        Returns
        -------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series or None
            Target variable (if create_target=True)
        """
        # Create features
        df_features = self.create_features(df)

        # Create target (shifted by forecast_horizon)
        if create_target and self.target_col in df_features.columns:
            y = df_features[self.target_col].shift(-self.forecast_horizon)

            # Remove rows where target is NaN
            valid_idx = ~y.isna()
            df_features = df_features[valid_idx]
            y = y[valid_idx]

            # Remove target from features
            X = df_features.drop(columns=[self.target_col])
        else:
            X = df_features
            if self.target_col in X.columns:
                X = X.drop(columns=[self.target_col])
            y = None

        # Remove any remaining NaN or inf values
        X = X.replace([np.inf, -np.inf], np.nan)

        # Store feature columns
        self.feature_columns_ = X.columns.tolist()

        return X, y

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the feature engine (mainly for scaler).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (after prepare_data)
        y : pd.Series, optional
            Target variable
        """
        # Fit scaler
        if self.feature_config.get('scaler_type') == 'standard':
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X)
        elif self.feature_config.get('scaler_type') == 'robust':
            self.scaler_ = RobustScaler()
            self.scaler_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler."""
        if self.scaler_ is not None:
            return self.scaler_.transform(X)
        return X.values

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X, y).transform(X)
