"""Statistical models for electricity price forecasting (ARIMA/SARIMAX)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List, Union
import warnings

# Import statsmodels for SARIMAX
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from pmdarima import auto_arima
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ArimaEstimator(BaseEstimator, RegressorMixin):
    """
    Wrapper for SARIMAX model to be compatible with scikit-learn pipeline.
    
    Supports using exogenous variables (X) in fit/predict (ARIMAX).
    Note: Time series indices must be continuous.
    
    Parameters
    ----------
    order : tuple, default=(1, 1, 1)
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters.
    seasonal_order : tuple, default=(0, 0, 0, 0)
        The (P,D,Q,s) order of the seasonal component.
    use_auto_arima : bool, default=False
        If True, uses pmdarima.auto_arima to find optimal order.
        WARNING: Very slow on large datasets.
    exog_cols : list, optional
        List of column names to use as exogenous variables.
        If None, uses all columns in X.
    scale_exog : bool, default=True
        Whether to scale exogenous variables before fitting.
    max_exog_size : int, default=10000
        Maximum number of samples to use for AutoARIMA fitting (most recent).
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        use_auto_arima: bool = False,
        exog_cols: Optional[List[str]] = None,
        scale_exog: bool = True,
        max_exog_size: int = 10000
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_auto_arima = use_auto_arima
        self.exog_cols = exog_cols
        self.scale_exog = scale_exog
        self.max_exog_size = max_exog_size
        
        self.model_ = None
        self.fitted_model_ = None
        self.scaler_ = None

        if not STATSMODELS_AVAILABLE:
            warnings.warn("Statsmodels/pmdarima not installed. ArimaEstimator will fail.")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the ARIMA/SARIMAX model.

        Parameters
        ----------
        X : pd.DataFrame
            Exogenous features.
        y : pd.Series
            Target time series.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Please install statsmodels and pmdarima.")

        # Prepare Exogenous Variables
        X_exog = self._prepare_exog(X)
        
        # Scaling
        if self.scale_exog and X_exog is not None:
            self.scaler_ = StandardScaler()
            X_exog = pd.DataFrame(
                self.scaler_.fit_transform(X_exog),
                columns=X_exog.columns,
                index=X_exog.index
            )

        # Auto ARIMA Logic
        if self.use_auto_arima:
            # truncate for speed if necessary
            if len(y) > self.max_exog_size:
                y_fit = y.iloc[-self.max_exog_size:]
                X_fit = X_exog.iloc[-self.max_exog_size:] if X_exog is not None else None
            else:
                y_fit = y
                X_fit = X_exog
                
            print(f"Running AutoARIMA on {len(y_fit)} samples...")
            self.model_ = auto_arima(
                y_fit,
                X=X_fit,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                m=24, # Daily seasonality (expensive!)
                seasonal=False, # Disable seasonal for speed initially
                d=1,
                trace=True,
                error_action='ignore',  
                suppress_warnings=True, 
                stepwise=True
            )
            self.order = self.model_.order
            self.seasonal_order = self.model_.seasonal_order
            self.fitted_model_ = self.model_ # auto_arima returns fitted model
            print(f"Best ARIMA Order: {self.order}")
            
        else:
            # Manual SARIMAX
            # We must use statsmodels directly
            self.model_ = SARIMAX(
                y,
                exog=X_exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model_ = self.model_.fit(disp=False)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict future values.
        
        For ARIMA, we need to forecast 'n' steps ahead.
        Scikit-learn 'predict' provides 'X' which implies the steps to predict 
        correspond to the rows of 'X'.
        """
        if self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted yet.")

        X_exog = self._prepare_exog(X)
        
        if self.scale_exog and self.scaler_ is not None and X_exog is not None:
             X_exog = pd.DataFrame(
                self.scaler_.transform(X_exog),
                columns=X_exog.columns,
                index=X_exog.index
            )
            
        # Determine number of steps (length of X)
        n_periods = len(X)
        
        # In statsmodels, we usually predict providing start/end or steps
        # If we trained on 0..T, we predict T+1..T+n using X_new
        
        # WARNING: This assumes X follows immediately after training data
        # If X is random validation set chunks, this will fail.
        # ARIMA is stateful.
        
        # For 'predict', ARIMAX needs the exogenous vars for the forecast horizon
        if self.use_auto_arima:
             preds = self.fitted_model_.predict(n_periods=n_periods, X=X_exog)
        else:
             # Using get_forecast for out-of-sample
             forecast = self.fitted_model_.get_forecast(steps=n_periods, exog=X_exog)
             preds = forecast.predicted_mean
             
        return preds.values

    def _prepare_exog(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Convert X to specific exogenous columns."""
        if X is None:
            return None
            
        if self.exog_cols:
            missing = [c for c in self.exog_cols if c not in X.columns]
            if missing:
                # Fallback or error?
                pass 
            return X[self.exog_cols].fillna(0)
            
        return X.fillna(0)
