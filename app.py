"""Streamlit Dashboard for Electricity Price Forecasting Model Comparison."""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

sys.path.append('src')

from utils.preprocessing import create_merged_dataset, split_train_val_test, handle_missing_values
from features.pipeline import TimeSeriesFeatureEngine
from models.train import load_model, evaluate_model
from models.baselines import NaivePersistence


@st.cache_data
def load_data():
    """Load and prepare data."""
    df = create_merged_dataset(db_path="data/cache.db")
    df_clean = handle_missing_values(df, strategy='forward_fill', limit=24).dropna()

    train_df, val_df, test_df = split_train_val_test(
        df_clean,
        train_end="2023-12-31",
        val_end="2024-06-30"
    )

    return df_clean, train_df, val_df, test_df


@st.cache_resource
def load_models():
    """Load all available models."""
    models = {}

    # Load XGBoost models
    model_dir = Path("models")

    for model_file in model_dir.glob("*.json"):
        model_name = model_file.stem

        # Skip feature files and config files (only load actual model files)
        if '_features' in model_name or 'params' in model_name:
            continue

        # Check if corresponding feature engine exists
        fe_path = model_dir / f"{model_name}_feature_engine.pkl"
        if not fe_path.exists():
            continue

        try:
            import xgboost as xgb
            model = xgb.XGBRegressor()

            # Fix for XGBoost _estimator_type error
            if not hasattr(model, "_estimator_type"):
                model._estimator_type = "regressor"

            model.load_model(str(model_file))

            # Load feature engine
            with open(fe_path, 'rb') as f:
                feature_engine = pickle.load(f)
            models[model_name] = {'model': model, 'feature_engine': feature_engine, 'type': 'xgboost'}
        except Exception as e:
            pass  # Silently skip files that can't be loaded

    return models


def prepare_features(df, feature_engine):
    """Prepare features using feature engine."""
    X, y = feature_engine.prepare_data(df, create_target=True)

    # Handle NaN values
    lag_cols = [col for col in X.columns if 'lag' in col.lower()]
    other_cols = [col for col in X.columns if col not in lag_cols]

    # Use ffill() instead of deprecated fillna(method='ffill')
    X[lag_cols] = X[lag_cols].ffill().fillna(0)
    X[other_cols] = X[other_cols].fillna(0)

    return X, y


def create_baseline_models(X_train, y_train):
    """Create baseline models."""
    baseline_24h = NaivePersistence(strategy='24h')
    baseline_24h.fit(X_train, y_train)

    baseline_168h = NaivePersistence(strategy='168h')
    baseline_168h.fit(X_train, y_train)

    return baseline_24h, baseline_168h


def plot_predictions(y_true, predictions_dict, start_idx=0, n_hours=168):
    """Plot predictions comparison."""
    end_idx = min(start_idx + n_hours, len(y_true))

    fig = go.Figure()

    # Actual prices
    fig.add_trace(go.Scatter(
        x=y_true.index[start_idx:end_idx],
        y=y_true.values[start_idx:end_idx],
        name='Actual',
        line=dict(color='black', width=2),
        mode='lines'
    ))

    # Predictions from each model
    colors = ['green', 'blue', 'red', 'orange', 'purple']
    for i, (name, pred) in enumerate(predictions_dict.items()):
        fig.add_trace(go.Scatter(
            x=y_true.index[start_idx:end_idx],
            y=pred[start_idx:end_idx],
            name=name,
            line=dict(color=colors[i % len(colors)], width=1.5),
            mode='lines'
        ))

    fig.update_layout(
        title='Model Predictions Comparison',
        xaxis_title='Date',
        yaxis_title='Price (€/MWh)',
        height=500,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


def plot_metrics_comparison(metrics_dict):
    """Plot metrics comparison bar chart."""
    metrics_df = pd.DataFrame(metrics_dict).T

    fig = go.Figure()

    for metric in ['MAE', 'RMSE']:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df.index,
            y=metrics_df[metric],
        ))

    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Error (€/MWh)',
        barmode='group',
        height=400
    )

    return fig


def plot_scatter(y_true, y_pred, model_name):
    """Plot predicted vs actual scatter plot."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true.values,
        y=y_pred,
        mode='markers',
        marker=dict(color='blue', size=4, opacity=0.5),
        name='Predictions'
    ))

    # Perfect prediction line
    min_val, max_val = y_true.min(), y_true.max()
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction'
    ))

    fig.update_layout(
        title=f'{model_name}: Predicted vs Actual',
        xaxis_title='Actual Price (€/MWh)',
        yaxis_title='Predicted Price (€/MWh)',
        height=500
    )

    return fig


def plot_error_distribution(errors, model_name):
    """Plot error distribution histogram."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=50,
        name='Errors',
        marker=dict(color='steelblue', line=dict(color='black', width=1))
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)

    fig.update_layout(
        title=f'{model_name}: Prediction Error Distribution',
        xaxis_title='Prediction Error (€/MWh)',
        yaxis_title='Frequency',
        height=400
    )

    return fig


def main():
    st.set_page_config(page_title="Electricity Price Forecasting", layout="wide")

    st.title("⚡ Electricity Price Forecasting Dashboard")
    st.markdown("Compare different models for 24-hour ahead electricity price forecasting in the Netherlands")

    # Load data
    with st.spinner("Loading data..."):
        df_clean, train_df, val_df, test_df = load_data()

    # Load models
    with st.spinner("Loading models..."):
        xgb_models = load_models()

    if not xgb_models:
        st.error("No models found in the models directory. Please train a model first.")
        return

    # Sidebar
    st.sidebar.header("Configuration")

    # Dataset selection
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset",
        ["Test", "Validation", "Train"]
    )

    dataset_map = {
        "Test": test_df,
        "Validation": val_df,
        "Train": train_df
    }
    selected_df = dataset_map[dataset_choice]

    # Model selection
    st.sidebar.subheader("Select Models to Compare")

    include_baseline_24h = st.sidebar.checkbox("Baseline 24h Persistence", value=True)
    include_baseline_168h = st.sidebar.checkbox("Baseline 168h Persistence", value=True)

    selected_xgb_models = []
    for model_name in xgb_models.keys():
        if st.sidebar.checkbox(f"XGBoost: {model_name}", value=True):
            selected_xgb_models.append(model_name)

    # Time range selection
    st.sidebar.subheader("Visualization Settings")
    n_hours = st.sidebar.slider("Hours to display", 24, 7*24, 7*24, 24)

    # Prepare features and get predictions
    all_predictions = {}
    all_metrics = {}
    y = None  # Will be set from first model's feature engine

    with st.spinner("Preparing features and generating predictions..."):
        # Baseline models - use first XGBoost model's feature engine for consistency
        if xgb_models and (include_baseline_24h or include_baseline_168h):
            first_model_name = list(xgb_models.keys())[0]
            baseline_fe = xgb_models[first_model_name]['feature_engine']

            # IMPORTANT: Create features on FULL dataset first, then split
            # This ensures lag features have access to historical data
            X_full, y_full = prepare_features(df_clean, baseline_fe)

            # Get date ranges for splitting
            train_end_date = pd.Timestamp("2023-12-31 23:00:00+01:00")
            val_end_date = pd.Timestamp("2024-06-30 23:00:00+02:00")

            # Split based on dataset choice
            if dataset_choice == "Train":
                X_baseline = X_full[X_full.index <= train_end_date]
                y = y_full[y_full.index <= train_end_date]
            elif dataset_choice == "Validation":
                X_baseline = X_full[(X_full.index > train_end_date) & (X_full.index <= val_end_date)]
                y = y_full[(y_full.index > train_end_date) & (y_full.index <= val_end_date)]
            else:  # Test
                X_baseline = X_full[X_full.index > val_end_date]
                y = y_full[y_full.index > val_end_date]

            # Get train data for baseline model fitting
            X_train_baseline = X_full[X_full.index <= train_end_date]
            y_train_baseline = y_full[y_full.index <= train_end_date]

            baseline_24h, baseline_168h = create_baseline_models(X_train_baseline, y_train_baseline)

            if include_baseline_24h:
                pred_24h = baseline_24h.predict(X_baseline)
                all_predictions['Baseline 24h'] = pred_24h
                all_metrics['Baseline 24h'] = {
                    'MAE': np.mean(np.abs(y.values - pred_24h)),
                    'RMSE': np.sqrt(np.mean((y.values - pred_24h)**2)),
                    'R2': 1 - np.sum((y.values - pred_24h)**2) / np.sum((y.values - np.mean(y.values))**2)
                }

            if include_baseline_168h:
                pred_168h = baseline_168h.predict(X_baseline)
                all_predictions['Baseline 168h'] = pred_168h
                all_metrics['Baseline 168h'] = {
                    'MAE': np.mean(np.abs(y.values - pred_168h)),
                    'RMSE': np.sqrt(np.mean((y.values - pred_168h)**2)),
                    'R2': 1 - np.sum((y.values - pred_168h)**2) / np.sum((y.values - np.mean(y.values))**2)
                }

        # XGBoost models - each uses its own feature engine
        for model_name in selected_xgb_models:
            model_data = xgb_models[model_name]
            model_fe = model_data['feature_engine']

            # IMPORTANT: Create features on FULL dataset first, then split
            X_full_model, y_full_model = prepare_features(df_clean, model_fe)

            # Split based on dataset choice
            if dataset_choice == "Train":
                X_model = X_full_model[X_full_model.index <= train_end_date]
                y_model = y_full_model[y_full_model.index <= train_end_date]
            elif dataset_choice == "Validation":
                X_model = X_full_model[(X_full_model.index > train_end_date) & (X_full_model.index <= val_end_date)]
                y_model = y_full_model[(y_full_model.index > train_end_date) & (y_full_model.index <= val_end_date)]
            else:  # Test
                X_model = X_full_model[X_full_model.index > val_end_date]
                y_model = y_full_model[y_full_model.index > val_end_date]

            # Scale features
            X_model_scaled = model_fe.transform(X_model)

            # Make predictions
            pred = model_data['model'].predict(X_model_scaled)
            all_predictions[model_name] = pred

            # Calculate metrics
            all_metrics[model_name] = {
                'MAE': np.mean(np.abs(y_model.values - pred)),
                'RMSE': np.sqrt(np.mean((y_model.values - pred)**2)),
                'R2': 1 - np.sum((y_model.values - pred)**2) / np.sum((y_model.values - np.mean(y_model.values))**2)
            }

            # Set y for plots (all models should have same target)
            if y is None:
                y = y_model

    # Display metrics
    st.header(f"📊 Performance Metrics ({dataset_choice} Set)")

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df = metrics_df.round(2)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best MAE", f"{metrics_df['MAE'].min():.2f} €/MWh",
                 delta=None, delta_color="inverse")
    with col2:
        st.metric("Best RMSE", f"{metrics_df['RMSE'].min():.2f} €/MWh",
                 delta=None, delta_color="inverse")
    with col3:
        st.metric("Best R²", f"{metrics_df['R2'].max():.4f}",
                 delta=None)

    st.dataframe(metrics_df.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
                                 .highlight_max(subset=['R2'], color='lightgreen'),
                use_container_width=True)

    # Metrics comparison chart
    st.plotly_chart(plot_metrics_comparison(all_metrics), use_container_width=True)

    # Predictions plot
    st.header("📈 Predictions Comparison")
    fig_pred = plot_predictions(y, all_predictions, 0, n_hours)
    st.plotly_chart(fig_pred, use_container_width=True)

    # Detailed model analysis
    st.header("🔍 Detailed Model Analysis")

    selected_model = st.selectbox(
        "Select a model for detailed analysis",
        list(all_predictions.keys())
    )

    if selected_model:
        y_pred = all_predictions[selected_model]
        errors = y.values - y_pred

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(plot_scatter(y, y_pred, selected_model), use_container_width=True)

        with col2:
            st.plotly_chart(plot_error_distribution(errors, selected_model), use_container_width=True)

        # Error analysis by hour
        st.subheader("Error Analysis by Hour of Day")
        error_by_hour = pd.DataFrame({
            'hour': y.index.hour,
            'error': np.abs(errors)
        }).groupby('hour')['error'].mean()

        fig_hour = go.Figure()
        fig_hour.add_trace(go.Bar(
            x=error_by_hour.index,
            y=error_by_hour.values,
            marker=dict(color='steelblue')
        ))
        fig_hour.update_layout(
            title='Mean Absolute Error by Hour of Day',
            xaxis_title='Hour of Day',
            yaxis_title='MAE (€/MWh)',
            height=400
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    # Feature importance (for XGBoost models only)
    if selected_model in selected_xgb_models:
        st.header("🎯 Feature Importance")
        model_data = xgb_models[selected_model]
        model_fe = model_data['feature_engine']
        importance = model_data['model'].feature_importances_

        if model_fe.feature_columns_ is not None:
            importance_df = pd.DataFrame({
                'feature': model_fe.feature_columns_,
                'importance': importance
            }).sort_values('importance', ascending=False).head(20)

            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                x=importance_df['importance'].values,
                y=importance_df['feature'].values,
                orientation='h',
                marker=dict(color='steelblue')
            ))
            fig_imp.update_layout(
                title=f'Top 20 Feature Importance - {selected_model}',
                xaxis_title='Importance',
                yaxis_title='Feature',
                height=600,
                yaxis=dict(autorange='reversed')
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # Dataset info
    with st.expander("ℹ️ Dataset Information"):
        st.write(f"**Dataset:** {dataset_choice}")
        st.write(f"**Date Range:** {selected_df.index.min()} to {selected_df.index.max()}")
        st.write(f"**Total Samples:** {len(selected_df)}")

        if y is not None:
            st.write(f"**Target Samples:** {len(y)}")

        # Show feature count for each model
        if selected_xgb_models:
            st.subheader("Model Configurations")
            for model_name in selected_xgb_models:
                model_fe = xgb_models[model_name]['feature_engine']
                if model_fe.feature_columns_ is not None:
                    st.write(f"**{model_name}:** {len(model_fe.feature_columns_)} features")

        st.subheader("Price Statistics")
        stats = selected_df['price'].describe()
        st.dataframe(stats)

    # Prediction diagnostics
    with st.expander("🔍 Prediction Diagnostics"):
        st.subheader("Prediction Statistics")
        for model_name, pred in all_predictions.items():
            st.write(f"**{model_name}:**")
            st.write(f"  - Mean: {np.mean(pred):.2f}, Std: {np.std(pred):.2f}")
            st.write(f"  - Min: {np.min(pred):.2f}, Max: {np.max(pred):.2f}")
            st.write(f"  - Unique values: {len(np.unique(pred))}")

        if 'Baseline 168h' in all_predictions and xgb_models:
            st.subheader("168h Baseline Debug Info")
            first_model_name = list(xgb_models.keys())[0]
            baseline_fe = xgb_models[first_model_name]['feature_engine']

            # Create features on full dataset, then get selected subset
            X_full_debug, _ = prepare_features(df_clean, baseline_fe)

            train_end_date = pd.Timestamp("2023-12-31 23:00:00+01:00")
            val_end_date = pd.Timestamp("2024-06-30 23:00:00+02:00")

            if dataset_choice == "Train":
                X_baseline_check = X_full_debug[X_full_debug.index <= train_end_date]
            elif dataset_choice == "Validation":
                X_baseline_check = X_full_debug[(X_full_debug.index > train_end_date) & (X_full_debug.index <= val_end_date)]
            else:  # Test
                X_baseline_check = X_full_debug[X_full_debug.index > val_end_date]

            if 'price_lag_144' in X_baseline_check.columns:
                lag_144 = X_baseline_check['price_lag_144']
                st.write(f"**price_lag_144 column:**")
                st.write(f"  - Mean: {lag_144.mean():.2f}, Std: {lag_144.std():.2f}")
                st.write(f"  - NaN count: {lag_144.isna().sum()} ({100*lag_144.isna().sum()/len(lag_144):.1f}%)")
                st.write(f"  - Min: {lag_144.min():.2f}, Max: {lag_144.max():.2f}")
            else:
                st.write("⚠️ price_lag_144 column not found in features!")


if __name__ == "__main__":
    main()
