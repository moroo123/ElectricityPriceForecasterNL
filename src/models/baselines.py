"""Baseline models for electricity price forecasting."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Dict, Optional, Union


class NaivePersistence(BaseEstimator, RegressorMixin):
    """
    Naive persistence model for time-series forecasting.

    For a 24-hour ahead forecast (predicting price at t+24 from features at time t):

    Strategies:
    - '24h': Daily persistence - "Tomorrow will be like today"
             Predicts price[t+24] = price[t]
             Uses price_lag_0 (current price at time t)

    - '168h': Weekly persistence - "Tomorrow will be like 7 days ago"
              Predicts price[t+24] = price[t-144] (6 days ago, same target hour)
              Uses price_lag_144 if available, otherwise price_lag_168 (7 days ago)

    Parameters
    ----------
    strategy : str, default='24h'
        Persistence strategy ('24h' or '168h')
    price_col : str, default='price'
        Name of the price column (if using raw data)
    """

    def __init__(self, strategy: str = '24h', price_col: str = 'price'):
        self.strategy = strategy
        self.price_col = price_col

    def fit(self, X, y=None):
        """Fit the model (no-op)."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using persistence logic.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Must contain specific lag columns.
            - For '24h' strategy: requires 'price_lag_0'
            - For '168h' strategy: requires 'price_lag_144' or 'price_lag_168'

        Returns
        -------
        np.ndarray
            Predictions
        """
        if self.strategy == '24h':
            # Use current price (lag 0) to predict 24h ahead
            if 'price_lag_0' in X.columns:
                return X['price_lag_0'].values
            else:
                raise ValueError("Feature 'price_lag_0' required for 24h strategy. "
                                 "Available lag features: " +
                                 str([col for col in X.columns if 'price_lag' in col]))

        elif self.strategy == '168h':
            # For weekly persistence predicting 24h ahead, ideally use lag 144 (6 days ago)
            # If not available, use lag 168 (7 days ago) as approximation
            if 'price_lag_144' in X.columns:
                return X['price_lag_144'].values
            elif 'price_lag_168' in X.columns:
                return X['price_lag_168'].values
            else:
                raise ValueError("Feature 'price_lag_144' or 'price_lag_168' required for 168h strategy. "
                                 "Available lag features: " +
                                 str([col for col in X.columns if 'price_lag' in col]))

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
