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
from finalproject.ml_models import SVRModel
from finalproject.evaluation import evaluate_model

# Define Location (site index) and load the input data of this location
site_index = 1
file_path = get_input_file_path(site_index)
df = pd.read_csv(file_path)


# Engineer features
df["Time"] = pd.to_datetime(df["Time"])
df = engineer_features(df)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2)

# Run SVR model
svr_model = SVRModel(train_df)
svr_model.train()
svr_preds = svr_model.predict(test_df)

# Evaluate the SVR model
mae_svr, mse_svr, rmse_svr = evaluate_model(svr_preds, test_df['Power'])
print(f"SVR MAE: {mae_svr:.4f}, MSE: {mse_svr:.4f}, RMSE: {rmse_svr:.4f}")


# Create a PersistenceModel instance and evaluate its performance
persistence_model = PersistenceModel(test_df)
persistence_preds = persistence_model.predict()
mae_persistence, mse_persistence, rmse_persistence = persistence_model.evaluate()
print(f"Persistence model MAE: {mae_persistence:.4f}, MSE: {mse_persistence:.4f}, RMSE: {rmse_persistence:.4f}")


# Plot time series for a specific variable and time period
variable_name = 'windspeed_100m'
starting_time = '2017-05-01 00:00:00'
ending_time = '2017-05-01 23:59:59'
fig, ax = plot_timeseries(variable_name, site_index, starting_time, ending_time)