"""Model training and evaluation for electricity price forecasting."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import pickle
import json

# ML libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# Feature engineering
import sys
sys.path.append(str(Path(__file__).parent.parent))

from features.pipeline import TimeSeriesFeatureEngine
from utils.preprocessing import (
    create_merged_dataset,
    split_train_val_test,
    handle_missing_values
)
from utils.logger import setup_logger
from .baselines import NaivePersistence
from .statistical import ArimaEstimator

logger = setup_logger("train_model")


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict] = None,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    verbose: bool = True
) -> xgb.XGBRegressor:
    """
    Train an XGBoost model for electricity price forecasting.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame, optional
        Validation features for early stopping
    y_val : pd.Series, optional
        Validation target for early stopping
    params : dict, optional
        XGBoost hyperparameters. If None, uses default parameters.
    n_estimators : int, default=1000
        Maximum number of boosting rounds
    early_stopping_rounds : int, default=50
        Early stopping rounds
    verbose : bool, default=True
        Whether to print training progress

    Returns
    -------
    xgb.XGBRegressor
        Trained XGBoost model
    """
    # Default hyperparameters optimized for electricity price forecasting
    # Tuned for better performance with time-series data
    default_params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 8,           # Increased to capture complex patterns
        'min_child_weight': 1,    # More flexible (was too conservative)
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'gamma': 0,               # Reduced pruning for better fit
        'reg_alpha': 0.01,        # Reduced L1 (was over-regularizing)
        'reg_lambda': 1.0,        # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
    }

    if params is not None:
        default_params.update(params)

    # Initialize model with early_stopping_rounds in constructor (XGBoost 2.0+)
    if X_val is not None and y_val is not None:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            **default_params
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            **default_params
        )

    # Prepare evaluation set for early stopping
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_train, y_train), (X_val, y_val)]

    # Train model
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=verbose
    )

    if verbose:
        logger.info(f"Best iteration: {model.best_iteration if eval_set else n_estimators}")
        logger.info(f"Best score: {model.best_score if eval_set else 'N/A'}")

    return model


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = '24h',
    verbose: bool = True
) -> NaivePersistence:
    """
    Train (initialize) a baseline model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (not used for fitting, but good for consistency)
    y_train : pd.Series
        Training target
    strategy : str, default='24h'
        '24h' (Daily Persistence) or '168h' (Weekly Persistence)
    verbose : bool, default=True
        Whether to print status
        
    Returns
    -------
    NaivePersistence
        Initialized baseline model
    """
    model = NaivePersistence(strategy=strategy)
    model.fit(X_train, y_train)
    
    if verbose:
        logger.info(f"Initialized Baseline Model (Strategy: {strategy})")
        
    return model


def train_statistical_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (0, 0, 0, 0),
    exog_cols: list = None,
    use_auto_arima: bool = False,
    verbose: bool = True
) -> ArimaEstimator:
    """
    Train an ARIMA/SARIMAX model.
    """
    model = ArimaEstimator(
        order=order,
        seasonal_order=seasonal_order,
        use_auto_arima=use_auto_arima,
        exog_cols=exog_cols
    )
    
    if verbose:
        logger.info(f"Training ARIMA Model (Auto: {use_auto_arima})...")
        
    model.fit(X_train, y_train)
    
    if verbose:
        logger.info("ARIMA training complete.")
        
    return model


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    set_name: str = "Test"
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained model
    X : pd.DataFrame
        Features
    y : pd.Series
        True values
    set_name : str
        Name of the dataset (for printing)

    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X)

    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    r2 = r2_score(y, y_pred)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

    logger.info(f"{set_name} Set Performance:")
    logger.info(f"  MAE:  {mae:.2f} €/MWh")
    logger.info(f"  RMSE: {rmse:.2f} €/MWh")
    logger.info(f"  MAPE: {mape:.2f}%")
    logger.info(f"  R²:   {r2:.4f}")

    return metrics


def get_feature_importance(
    model: xgb.XGBRegressor,
    feature_names: list,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Get feature importance from trained model.

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained model
    feature_names : list
        List of feature names
    top_n : int, default=20
        Number of top features to return

    Returns
    -------
    pd.DataFrame
        DataFrame with feature importance scores
    """
    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return importance_df.head(top_n)


def save_model(
    model,
    feature_engine: TimeSeriesFeatureEngine,
    save_dir: str = "models",
    model_name: str = "xgboost_price_forecast"
):
    """
    Save trained model and feature engine.

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained model
    feature_engine : TimeSeriesFeatureEngine
        Fitted feature engine
    save_dir : str
        Directory to save model
    model_name : str
        Base name for saved files
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    # Save XGBoost model
    model_path = save_path / f"{model_name}.json"
    
    # Patch for XGBoost < 2.1 issue where _estimator_type might be missing
    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "regressor"
        
    model.save_model(str(model_path))
    logger.info(f"Model saved to: {model_path}")

    # Save feature engine
    feature_engine_path = save_path / f"{model_name}_feature_engine.pkl"
    with open(feature_engine_path, 'wb') as f:
        pickle.dump(feature_engine, f)
    logger.info(f"Feature engine saved to: {feature_engine_path}")

    # Save feature names
    if feature_engine.feature_columns_ is not None:
        features_path = save_path / f"{model_name}_features.json"
        with open(features_path, 'w') as f:
            json.dump(feature_engine.feature_columns_, f, indent=2)
        logger.info(f"Feature names saved to: {features_path}")


def load_model(
    save_dir: str = "models",
    model_name: str = "xgboost_price_forecast"
) -> Tuple:
    """
    Load trained model and feature engine.

    Parameters
    ----------
    save_dir : str
        Directory where model is saved
    model_name : str
        Base name of saved files

    Returns
    -------
    model : xgb.XGBRegressor
        Loaded model
    feature_engine : TimeSeriesFeatureEngine
        Loaded feature engine
    """
    save_path = Path(save_dir)

    # Load XGBoost model
    model = xgb.XGBRegressor()
    model_path = save_path / f"{model_name}.json"
    model.load_model(str(model_path))
    logger.info(f"Model loaded from: {model_path}")

    # Load feature engine
    feature_engine_path = save_path / f"{model_name}_feature_engine.pkl"
    with open(feature_engine_path, 'rb') as f:
        feature_engine = pickle.load(f)
    logger.info(f"Feature engine loaded from: {feature_engine_path}")

    return model, feature_engine


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    params: Optional[Dict] = None,
    verbose: bool = True
) -> Dict[str, list]:
    """
    Perform time-series cross-validation.

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    n_splits : int, default=5
        Number of CV splits
    params : dict, optional
        Model hyperparameters
    verbose : bool, default=True
        Whether to print progress

    Returns
    -------
    dict
        Dictionary of CV scores for each metric
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_scores = {
        'MAE': [],
        'RMSE': [],
        'MAPE': [],
        'R2': []
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        if verbose:
            logger.info(f"Fold {fold}/{n_splits}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train model
        model = train_xgboost_model(
            X_train, y_train,
            X_val, y_val,
            params=params,
            verbose=False
        )

        # Evaluate
        metrics = evaluate_model(model, X_val, y_val, set_name=f"Fold {fold}")

        for key in cv_scores.keys():
            cv_scores[key].append(metrics[key])

    # Print summary
    if verbose:
        logger.info("="*50)
        logger.info("Cross-Validation Summary:")
        logger.info("="*50)
        for metric, scores in cv_scores.items():
            logger.info(f"{metric}: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

    return cv_scores


def predict_future(
    model,
    feature_engine: TimeSeriesFeatureEngine,
    data: pd.DataFrame,
    forecast_horizon: int = 24
) -> pd.Series:
    """
    Make future predictions.

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained model
    feature_engine : TimeSeriesFeatureEngine
        Fitted feature engine
    data : pd.DataFrame
        Recent data for creating features
    forecast_horizon : int
        Hours ahead to forecast

    Returns
    -------
    pd.Series
        Predictions
    """
    # Create features (without target)
    X, _ = feature_engine.prepare_data(data, create_target=False)

    # Scale features
    X_scaled = feature_engine.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)

    # Create series with appropriate timestamps
    pred_index = X.index + pd.Timedelta(hours=forecast_horizon)
    pred_series = pd.Series(predictions, index=pred_index, name='price_forecast')

    return pred_series
