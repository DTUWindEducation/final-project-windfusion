import pytest
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from finalproject import evaluate_model

def test_evaluate_model():
    """Test the evaluate_model function with various scenarios."""
    
    # Test case 1: Perfect prediction
    predictions = np.array([1, 2, 3, 4, 5])
    actuals = np.array([1, 2, 3, 4, 5])
    mae, mse, rmse = evaluate_model(predictions, actuals)
    assert mae == 0
    assert mse == 0
    assert rmse == 0
    
    # Test case 2: Constant offset
    predictions = np.array([2, 3, 4, 5, 6])
    actuals = np.array([1, 2, 3, 4, 5])
    mae, mse, rmse = evaluate_model(predictions, actuals)
    assert mae == 1
    assert mse == 1
    assert rmse == 1
    
    # Test case 3: Random errors
    predictions = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    actuals = np.array([1, 2, 3, 4, 5])
    mae, mse, rmse = evaluate_model(predictions, actuals)
    
    # Calculate expected values
    expected_mae = mean_absolute_error(actuals, predictions)
    expected_mse = mean_squared_error(actuals, predictions)
    expected_rmse = np.sqrt(expected_mse)
    
    assert mae == pytest.approx(expected_mae)
    assert mse == pytest.approx(expected_mse)
    assert rmse == pytest.approx(expected_rmse)
    
    # Test case 4: Empty arrays
    with pytest.raises(ValueError):
        evaluate_model(np.array([]), np.array([]))
    
    # Test case 5: Different length arrays
    with pytest.raises(ValueError):
        evaluate_model(np.array([1, 2]), np.array([1]))
        
    # Test case 6: Single value
    predictions = np.array([5])
    actuals = np.array([4])
    mae, mse, rmse = evaluate_model(predictions, actuals)
    assert mae == 1
    assert mse == 1
    assert rmse == 1