"""
This module implements the PersistenceModel class, which provides
a simple persistence-based prediction model for time series data.
"""

import numpy as np
import pandas as pd


class PersistenceModel: 
    """
    A persistence model for time series data that predicts the next value
    based on the previous value.
    """
    
    def __init__(self, df, target_column='Power'):
        """
        Initialize the PersistenceModel with a pandas DataFrame.

        Parameters:
        data (pd.DataFrame): The input data for the model.
        """
        self.df = df
        self.target = target_column
        self.preds = None
        self.true = None


    def predict(self):
        """
        Predict one hour ahead values using the persistence model
        """
        self.preds = self.df[self.target].shift(1).iloc[1:]
        self.true = self.df[self.target].iloc[1:]
        return self.preds
    

    def evaluate(self):
        """
        Evaluate the model using Mean Absolute Error (MAE),
        Mean Squared Error (MSE), and Root Mean Squared Error (RMSE)
        """
        if self.preds is None or self.true is None:
            raise ValueError("You must call predict() before evaluate()")
        
        mae = np.mean(np.abs(self.preds - self.true))
        mse = np.mean((self.preds - self.true) ** 2)
        rmse = np.sqrt(mse)
        return mae, mse, rmse
