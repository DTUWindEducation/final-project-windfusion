import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import NotFittedError

# Assuming the SVRModel class is imported from your module
from finalproject import SVRModel, train_test_split

def test_svr_model():
    """Test function for SVRModel class."""
    
    # Create synthetic regression data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
    target_col = 'target'
    
    # Create DataFrame with Time column
    df = pd.DataFrame(X, columns=feature_cols)
    df[target_col] = y
    df['Time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2)
    
    # Test initialization
    model = SVRModel(train_df, target_col=target_col)
    
    # Check initialization values
    assert model.target_col == target_col
    assert isinstance(model.features, list)
    assert len(model.features) == len(feature_cols)
    assert 'Time' not in model.features
    assert target_col not in model.features
    
    # Test training
    model.train()  # Should not raise any exceptions
    
    # Test prediction
    test_predictions = model.predict(test_df)
    
    # Check prediction output
    assert isinstance(test_predictions, np.ndarray)
    assert len(test_predictions) == len(test_df)
    assert not np.isnan(test_predictions).any()
    
    # Test with missing features (should raise KeyError)
    test_df_missing = test_df.drop(columns=[model.features[0]])
    with pytest.raises(KeyError):
        model.predict(test_df_missing)
    
    # Test with empty DataFrame (should raise ValueError)
    empty_df = pd.DataFrame(columns=df.columns)
    with pytest.raises(ValueError):
        model.predict(empty_df)
    
    # Test prediction before training (should raise NotFittedError)
    untrained_model = SVRModel(train_df, target_col=target_col)
    with pytest.raises(NotFittedError):
        untrained_model.predict(test_df)
    
    # Test performance (optional - just a sanity check)
    train_predictions = model.predict(train_df)
    train_rmse = np.sqrt(mean_squared_error(train_df[target_col], train_predictions))
    assert train_rmse < np.std(train_df[target_col])  # Model should be better than naive prediction