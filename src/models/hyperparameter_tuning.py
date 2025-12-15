"""Hyperparameter tuning using Optuna for electricity price forecasting."""

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)


def objective_xgboost(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> float:
    """
    Optuna objective function for XGBoost hyperparameter tuning.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target

    Returns
    -------
    float
        Validation RMSE (to minimize)
    """
    # Suggest hyperparameters (Regularized Search Space)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1,

        # Learning rate - slightly higher start to allow simple models to learn
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),

        # Tree structure - Restricted to prevent memorization
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        
        # Min child weight - INCREASED significantly to force generalization
        # No leaf can exist with less than 10-100 samples
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 100),

        # Sampling - Encourage bagging
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.9),

        # Regularization - Stronger priors
        'gamma': trial.suggest_float('gamma', 0.1, 5.0),  # Minimum loss reduction
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),  # L1
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),  # L2
    }

    # Number of estimators
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)

    # Train model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        early_stopping_rounds=50,
        **params
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predict and calculate RMSE
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    return rmse


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
    timeout: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Dict, optuna.Study]:
    """
    Tune XGBoost hyperparameters using Optuna.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
    n_trials : int, default=100
        Number of optimization trials
    timeout : int, optional
        Timeout in seconds (useful for time-limited tuning)
    verbose : bool, default=True
        Whether to print progress

    Returns
    -------
    best_params : dict
        Best hyperparameters found
    study : optuna.Study
        Optuna study object (contains optimization history)

    Example
    -------
    >>> best_params, study = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=50)
    >>> # Train final model with best params
    >>> model = train_xgboost_model(X_train, y_train, X_val, y_val, params=best_params)
    """
    if verbose:
        print("="*60)
        print("HYPERPARAMETER TUNING WITH OPTUNA")
        print("="*60)
        print(f"Number of trials: {n_trials}")
        if timeout:
            print(f"Timeout: {timeout} seconds")
        print()

    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        study_name='xgboost_price_forecasting'
    )

    # Optimize
    study.optimize(
        lambda trial: objective_xgboost(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose
    )

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    if verbose:
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Best RMSE: {best_value:.2f} €/MWh")
        print(f"\nBest parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

    return best_params, study


def plot_optimization_history(study: optuna.Study):
    """
    Plot Optuna optimization history.

    Parameters
    ----------
    study : optuna.Study
        Optuna study object

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure
    """
    try:
        import plotly.graph_objects as go

        # Get trials
        trials = study.trials
        values = [trial.value for trial in trials]

        # Create plot
        fig = go.Figure()

        # Optimization history
        fig.add_trace(go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode='lines+markers',
            name='Trial RMSE',
            line=dict(color='steelblue'),
            marker=dict(size=4)
        ))

        # Best value line
        best_values = [min(values[:i+1]) for i in range(len(values))]
        fig.add_trace(go.Scatter(
            x=list(range(len(best_values))),
            y=best_values,
            mode='lines',
            name='Best RMSE',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title='Hyperparameter Optimization History',
            xaxis_title='Trial',
            yaxis_title='Validation RMSE (€/MWh)',
            height=400,
            showlegend=True
        )

        return fig
    except ImportError:
        print("Plotly not available. Install it to visualize optimization history.")
        return None


def plot_param_importance(study: optuna.Study):
    """
    Plot parameter importance.

    Parameters
    ----------
    study : optuna.Study
        Optuna study object

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure
    """
    try:
        import plotly.graph_objects as go
        from optuna.importance import get_param_importances

        # Get parameter importances
        importances = get_param_importances(study)

        # Sort by importance
        params = list(importances.keys())
        values = [importances[p] for p in params]

        # Create plot
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=values,
            y=params,
            orientation='h',
            marker=dict(color='steelblue')
        ))

        fig.update_layout(
            title='Hyperparameter Importance',
            xaxis_title='Importance',
            yaxis_title='Parameter',
            height=500,
            yaxis=dict(autorange='reversed')
        )

        return fig
    except ImportError:
        print("Required packages not available.")
        return None


def quick_tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 30
) -> Dict:
    """
    Quick hyperparameter tuning with fewer trials.

    Good for fast iteration and getting a reasonable baseline.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    n_trials : int, default=30
        Number of trials (30 is usually enough for decent results)

    Returns
    -------
    best_params : dict
        Best hyperparameters found
    """
    print(f"Running quick tuning with {n_trials} trials...")
    best_params, _ = tune_xgboost(
        X_train, y_train, X_val, y_val,
        n_trials=n_trials,
        verbose=True
    )
    return best_params


def extensive_tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    timeout_minutes: int = 60
) -> Tuple[Dict, optuna.Study]:
    """
    Extensive hyperparameter tuning with time limit.

    Runs as many trials as possible within the time limit.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    timeout_minutes : int, default=60
        Maximum time for tuning in minutes

    Returns
    -------
    best_params : dict
        Best hyperparameters found
    study : optuna.Study
        Study object for analysis
    """
    timeout_seconds = timeout_minutes * 60
    print(f"Running extensive tuning with {timeout_minutes} minute timeout...")

    best_params, study = tune_xgboost(
        X_train, y_train, X_val, y_val,
        n_trials=1000,  # High number, will be limited by timeout
        timeout=timeout_seconds,
        verbose=True
    )

    print(f"\nCompleted {len(study.trials)} trials in {timeout_minutes} minutes")
    return best_params, study
