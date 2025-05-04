import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import NotFittedError
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# Assuming the RandomForestModel is imported from your module
from finalproject import RandomForestModel, train_test_split

def test_random_forest_model_initialization():
    """Test initialization of RandomForestModel."""
    # Create synthetic data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['Power'] = y
    df['Time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
    
    # Test default initialization
    model = RandomForestModel(df)
    assert model.target_col == 'Power'
    assert len(model.features) == 5  # Should exclude 'Time' and 'Power'
    assert 'Time' not in model.features
    assert 'Power' not in model.features
    assert model.model.n_estimators == 100
    assert model.model.random_state == 42
    
    # Test with custom target column
    df['custom_target'] = y
    custom_model = RandomForestModel(df, target_col='custom_target')
    assert custom_model.target_col == 'custom_target'
    assert 'custom_target' not in custom_model.features

def test_random_forest_model_training_and_prediction():
    """Test training and prediction functionality."""
    # Create synthetic data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['Power'] = y
    df['Time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2)
    
    # Initialize and train model
    model = RandomForestModel(train_df)
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

def test_random_forest_model_error_handling():
    """Test error cases and edge cases."""
    # Create synthetic data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['Power'] = y
    df['Time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
    
    train_df, test_df = train_test_split(df, test_size=0.2)
    model = RandomForestModel(train_df)
    
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
    
def test_random_forest_feature_importance():
    """Test that feature importance can be accessed after training."""
    # Create synthetic data with one important feature
    X = np.random.rand(100, 3)
    y = X[:, 0] * 10 + np.random.normal(0, 0.1, 100)  # First feature is important
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
    df['Power'] = y
    df['Time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
    
    model = RandomForestModel(df)
    model.train()
    
    # Check feature importance
    importances = model.model.feature_importances_
    assert len(importances) == len(model.features)
    assert importances[0] > importances[1]  # First feature should be more important
    assert importances[0] > importances[2]

def test_random_forest_reproducibility():
    """Test that model produces same results with fixed random state."""
    # Create synthetic data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['Power'] = y
    df['Time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
    
    train_df, test_df = train_test_split(df, test_size=0.2)
    
    # Train two models with same random state
    model1 = RandomForestModel(train_df)
    model1.train()
    pred1 = model1.predict(test_df)
    
    model2 = RandomForestModel(train_df)
    model2.train()
    pred2 = model2.predict(test_df)
    
    # Predictions should be identical with fixed random state
    assert np.array_equal(pred1, pred2)

def test_random_forest_with_single_feature():
    """Test model works with single feature."""
    # Create data with single feature
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'Power': np.random.rand(100),
        'Time': pd.date_range(start='2023-01-01', periods=100, freq='h')
    })
    
    model = RandomForestModel(df)
    model.train()
    predictions = model.predict(df)
    assert len(predictions) == len(df)