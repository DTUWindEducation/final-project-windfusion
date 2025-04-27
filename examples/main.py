import numpy as np
import pandas as pd
import sys, os

# Dynamically add src/ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from finalproject.persistence_model import PersistenceModel
from finalproject.path_utils import get_input_file_path
from finalproject.feature_engineer import engineer_features
from finalproject import plot_timeseries
from finalproject.split_data import train_test_split
from finalproject.ml_models import SVRModel, XGBModel, LSTMModel
from finalproject.evaluation import evaluate_model
from finalproject.plotting_predicted_vs_real import plot_power_predictions

# Define Location (site index) and load the input data of this location
site_index = 1
ML_model = 'LSTM'  # can also be 'XGB', or 'LSTM'
file_path = get_input_file_path(site_index)
df = pd.read_csv(file_path)

# Engineer features
df["Time"] = pd.to_datetime(df["Time"])
df = engineer_features(df)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2)
actual_values = test_df['Power']
timestamps = test_df["Time"]

# Run the selected ML model based on ML_model variable
if ML_model == 'SVR':
    model = SVRModel(train_df)
    model.train()
    predictions = model.predict(test_df)
    model_name = "SVR"
    # Evaluate the model
    mae, mse, rmse = evaluate_model(predictions, test_df['Power'])
elif ML_model == 'XGB':
    model = XGBModel(train_df)
    model.train()
    predictions = model.predict(test_df)
    model_name = "XGB"
    # Evaluate the model
    mae, mse, rmse = evaluate_model(predictions, test_df['Power'])
elif ML_model == 'LSTM':
    model = LSTMModel(train_df)
    model.train()
    predictions = model.predict(test_df)
    model_name = "LSTM"
    # For LSTM, adjust test data to match predictions length
    actual_values = test_df['Power'].iloc[model.time_steps:].reset_index(drop=True)
    timestamps = test_df["Time"].iloc[model.time_steps:].reset_index(drop=True)
    # Ensure predictions and actual values have the same length
    if len(predictions) < len(actual_values):
        actual_values = actual_values[:len(predictions)]
    elif len(actual_values) < len(predictions):
        predictions = predictions[:len(actual_values)]
    # Evaluate the model
    mae, mse, rmse = evaluate_model(predictions, actual_values)
else:
    raise ValueError(f"Unknown ML model: {ML_model}. Choose from 'SVR', 'XGB', or 'LSTM'.")

print(f"{model_name} MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

# Create a PersistenceModel instance and evaluate its performance
persistence_model = PersistenceModel(test_df)
persistence_preds = persistence_model.predict()
mae_persistence, mse_persistence, rmse_persistence = persistence_model.evaluate()
print(f"Persistence model MAE: {mae_persistence:.4f}, MSE: {mse_persistence:.4f}, RMSE: {rmse_persistence:.4f}")

# Plot predictions for a specific period
time_window = ('2021-10-01 00:00:00', '2021-11-01 23:59:59')
fig2, ax2 = plot_power_predictions(site_index, timestamps, predictions, actual_values, ML_model, time_window=time_window)

# Plot time series for a specific variable and time period
variable_name = 'windspeed_100m'
starting_time = '2017-05-01 00:00:00'
ending_time = '2017-05-01 23:59:59'
fig, ax = plot_timeseries(variable_name, site_index, starting_time, ending_time)