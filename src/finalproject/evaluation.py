from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(predictions, actuals):
    """
    Evaluate model predictions using MAE, MSE, and RMSE.

    Parameters
    ----------
    predictions : array-like
        Predicted values from the model.
    actuals : array-like
        Ground truth values to compare against.

    Returns
    -------
    mae : float
        Mean Absolute Error
    mse : float
        Mean Squared Error
    rmse : float
        Root Mean Squared Error
    """
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    return mae, mse, rmse
