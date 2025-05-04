import numpy as np
import pandas as pd
import sys, os

# Add src/ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from finalproject import *

# Define Location (site index) and machine learning model 
site_index = 1
ML_model = 'RandomForest'  # ['SVR', 'GradientBoosting', 'RandomForest', 'FeedforwardNN']

# Define time window for predictions vs true values plot
time_window = ('2021-10-01 00:00:00', '2021-11-01 23:59:59')

# Define a specific variable and time period to plot time series
variable_name = 'windspeed_100m'
starting_time = '2017-05-01 00:00:00'
ending_time = '2017-05-01 23:59:59'

# Create and print site summary
site_summary = SiteSummary(site_index)
summary_info = site_summary.summarize()

# Load the input data of this location
file_path = get_input_file_path(site_index)
df = pd.read_csv(file_path)

# Engineer features (transform the data)
df["Time"] = pd.to_datetime(df["Time"])
df = engineer_features(df)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2)
actual_values = test_df['Power']
timestamps = test_df["Time"]

# Run the selected ML model based on ML_model variable
if ML_model == 'SVR':
    model = SVRModel(train_df)
elif ML_model == 'GradientBoosting':
    model = GradientBoostingModel(train_df)
elif ML_model == 'RandomForest':
    model = RandomForestModel(train_df)
elif ML_model == 'FeedforwardNN':
    model = FeedforwardNNModel(train_df)
else:
    raise ValueError(f"Unknown ML model: {ML_model}. Choose from 'SVR', 'GradientBoosting', 'LagLinear', or 'FeedforwardNN'.")

# Train the model and make predictions
model.train()
predictions = model.predict(test_df)
# Evaluate the model
mae, mse, rmse = evaluate_model(predictions, actual_values)
print(f"{ML_model} MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

# Create a PersistenceModel instance and evaluate its performance
persistence_model = PersistenceModel(test_df)
persistence_preds = persistence_model.predict()
mae_persistence, mse_persistence, rmse_persistence = persistence_model.evaluate()
print(f"Persistence model MAE: {mae_persistence:.4f}, MSE: {mse_persistence:.4f}, RMSE: {rmse_persistence:.4f}")

# Plot power predictions
fig2, ax2 = plot_power_predictions(site_index, timestamps, predictions, actual_values, ML_model, time_window=time_window)

# Save prediction plot
save_figure(fig2, f"{ML_model}_predictions_site_{site_index}.png")

# Plot and save time series
fig, ax = plot_timeseries(variable_name, site_index, starting_time, ending_time)
save_figure(fig, f"timeseries_{variable_name}_site_{site_index}.png")

# Plot feature importance and save the plot
if hasattr(model.model, "feature_importances_"):
    fig_feat, ax_feat = plot_feature_importance(model.model, model.features)
    save_figure(fig_feat, f"{ML_model}_feature_importance_site_{site_index}.png")

# Evaluate the model without power lag1
evaluate_model_without_power_lag1(df, ML_model, site_index)
