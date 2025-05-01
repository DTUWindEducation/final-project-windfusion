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
from finalproject.ml_models import SVRModel, GradientBoostingModel, LagLinearModel, FeedforwardNNModel
from finalproject.evaluation import evaluate_model
from finalproject.plotting_predicted_vs_real import plot_power_predictions

# Define Location (site index) and load the input data of this location
site_index = 1
ML_model = 'FeedforwardNN'  # ['SVR', 'GradientBoosting', 'LagLinear', 'FeedforwardNN']
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
elif ML_model == 'GradientBoosting':
    model = GradientBoostingModel(train_df)
    model.train()
    predictions = model.predict(test_df)
    model_name = "GradientBoosting"
    # Evaluate the model
    mae, mse, rmse = evaluate_model(predictions, test_df['Power'])
elif ML_model == 'LagLinear':
    model = LagLinearModel(train_df)
    model.train()
    predictions = model.predict(test_df)
    actual_values = test_df['Power'].iloc[model.lags:].reset_index(drop=True)  # skip first 24 rows to align
    timestamps = test_df["Time"].iloc[model.lags:].reset_index(drop=True)
    model_name = "LagLineaR"
    # Evaluate the model
    mae, mse, rmse = evaluate_model(predictions, actual_values)
elif ML_model == 'FeedforwardNN':
    model = FeedforwardNNModel(train_df)
    model.train()
    predictions = model.predict(test_df)
    model_name = "FeedforwardNN"
    # Evaluate the model
    mae, mse, rmse = evaluate_model(predictions, test_df['Power'])
else:
    raise ValueError(f"Unknown ML model: {ML_model}. Choose from 'SVR', 'GradientBoosting', 'LagLinear', or 'FeedforwardNN'.")

print(f"{model_name} MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

# Create a PersistenceModel instance and evaluate its performance
persistence_model = PersistenceModel(test_df)
persistence_preds = persistence_model.predict()
mae_persistence, mse_persistence, rmse_persistence = persistence_model.evaluate()
print(f"Persistence model MAE: {mae_persistence:.4f}, MSE: {mse_persistence:.4f}, RMSE: {rmse_persistence:.4f}")

# Plot predictions for a specific period
time_window = ('2021-10-01 00:00:00', '2021-11-01 23:59:59')
fig2, ax2 = plot_power_predictions(site_index, timestamps, predictions, actual_values, ML_model, time_window=time_window)

# Save the predictions plot
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Go up one level from examples/
output_dir = os.path.join(project_root, 'outputs')
os.makedirs(output_dir, exist_ok=True)
plot_filename = f"{ML_model}_predictions_site_{site_index}.png"
plot_path = os.path.join(output_dir, plot_filename)
fig2.savefig(plot_path)
print(f"Predictions plot saved to: {plot_path}")

# Plot time series for a specific variable and time period
variable_name = 'windspeed_100m'
starting_time = '2017-05-01 00:00:00'
ending_time = '2017-05-01 23:59:59'
fig, ax = plot_timeseries(variable_name, site_index, starting_time, ending_time)

# Save the time series plot
timeseries_filename = f"timeseries_{variable_name}_site_{site_index}.png"
timeseries_path = os.path.join(output_dir, timeseries_filename)
fig.savefig(timeseries_path)
print(f"Time series plot saved to: {timeseries_path}")