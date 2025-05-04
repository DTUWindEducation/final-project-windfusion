import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import NotFittedError
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# Assuming the GradientBoostingModel is imported from your module
from finalproject import GradientBoostingModel, train_test_split

def test_gradient_boosting_model_initialization():
    """Test initialization of GradientBoostingModel."""
    # Create synthetic data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['Power'] = y
    df['Time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
    
    # Test default initialization
    model = GradientBoostingModel(df)
    assert model.target_col == 'Power'
    assert len(model.features) == 5  # Should exclude 'Time' and 'Power'
    assert 'Time' not in model.features
    assert 'Power' not in model.features
    assert model.model.n_estimators == 100
    assert model.model.learning_rate == 0.1
    assert model.model.max_depth == 3
    
    # Test custom initialization
    custom_model = GradientBoostingModel(
        df,
        target_col='Power',
        n_estimators=50,
        learning_rate=0.05,
        max_depth=5
    )
    assert custom_model.model.n_estimators == 50
    assert custom_model.model.learning_rate == 0.05
    assert custom_model.model.max_depth == 5

def test_gradient_boosting_model_training_and_prediction():
    """Test training and prediction functionality."""
    # Create synthetic data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['Power'] = y
    df['Time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2)
    
    # Initialize and train model
    model = GradientBoostingModel(train_df)
    model.train()  # Should not raise exceptions
    
    # Test predictions
    predictions = model.predict(test_df)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(test_df)
    assert not np.isnan(predictions).any()
    
    # Test performance (basic sanity check)
    train_predictions = model.predict(train_df)
    train_rmse = np.sqrt(mean_squared_error(train_df['Power'], train_predictions))
    assert train_rmse < np.std(train_df['Power'])  # Model should be better than naive prediction

def test_gradient_boosting_model_error_handling():
    """Test error cases and edge cases."""
    # Create synthetic data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['Power'] = y
    df['Time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
    
    train_df, test_df = train_test_split(df, test_size=0.2)
    model = GradientBoostingModel(train_df)
    
    # Test prediction before training
    with pytest.raises(NotFittedError):
        model.predict(test_df)
    
    # Train the model
    model.train()
    
    # Test with missing features
    test_df_missing = test_df.drop(columns=[model.features[0]])
    with pytest.raises(KeyError):
        model.predict(test_df_missing)
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame(columns=df.columns)
    with pytest.raises(ValueError):
        model.predict(empty_df)
    
    # Test with completely NaN data
    nan_df = test_df.copy()
    nan_df[model.features] = np.nan
    with pytest.raises(ValueError):
        model.predict(nan_df)

def test_gradient_boosting_model_with_custom_parameters():
    """Test model with custom hyperparameters."""
    # Create synthetic data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['Power'] = y
    df['Time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
    
    train_df, test_df = train_test_split(df, test_size=0.2)
    
    # Test with custom parameters
    custom_model = GradientBoostingModel(
        train_df,
        n_estimators=50,
        learning_rate=0.05,
        max_depth=5
    )
    custom_model.train()
    predictions = custom_model.predict(test_df)
    
    assert len(predictions) == len(test_df)
    assert custom_model.model.n_estimators == 50
    assert custom_model.model.learning_rate == 0.05
    assert custom_model.model.max_depth == 5

def test_feature_scaling_effect():
    """Verify that feature scaling is being applied correctly."""
    # Create data with different scales
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 100, 100),
        'Power': np.random.normal(0, 1, 100),
        'Time': pd.date_range(start='2023-01-01', periods=100, freq='h')
    })
    
    train_df, test_df = train_test_split(df, test_size=0.2)
    model = GradientBoostingModel(train_df)
    model.train()
    
    # Check that scaler was fitted
    assert hasattr(model.scaler, 'mean_')
    assert hasattr(model.scaler, 'scale_')
    
    # Verify scaling was applied by checking feature means are close to 0
    X_scaled = model.scaler.transform(train_df[model.features])
    assert np.allclose(X_scaled.mean(axis=0), np.zeros(len(model.features)), atol=1e-7)
    assert np.allclose(X_scaled.std(axis=0), np.ones(len(model.features)), atol=1e-7)