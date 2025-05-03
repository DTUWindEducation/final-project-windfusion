"""
This module provides utilities for wind power forecasting models, evaluation,
visualization, and data preprocessing used in the WindFusion project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


def load_observations_data(file_path):
    """
    Load and parse observations dataset from a full file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    df = pd.read_csv(file_path, engine='python')
    basic_stats = df.describe(include='all').round(2)
    return df, basic_stats


def plot_timeseries(variable_name, site_index, starting_time, ending_time):
    """
    Plot a time series of a selected variable for a given site
    within a specific period.
    """
    if variable_name not in ['windspeed_100m', 'Power']:
        raise ValueError("variable_name must be either 'windspeed_100m' or 'Power'")

    if site_index not in [1, 2, 3, 4]:
        raise ValueError("site_index must be 1, 2, 3, or 4")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    inputs_folder = os.path.join(project_root, 'inputs')
    data_path = os.path.join(inputs_folder, f"Location{site_index}.csv")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file {data_path} not found.") from e

    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
    else:
        raise ValueError("Time column not found in the data")

    if variable_name not in df.columns:
        raise ValueError(f"Variable '{variable_name}' not found in the data")

    start_dt = pd.to_datetime(starting_time)
    end_dt = pd.to_datetime(ending_time)
    filtered_df = df[(df['Time'] >= start_dt) & (df['Time'] <= end_dt)]

    if filtered_df.empty:
        raise ValueError("No data found for the specified time period")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(filtered_df['Time'], filtered_df[variable_name], 'b-', linewidth=1.5)
    ax.set_xticks(filtered_df['Time'][::max(1, len(filtered_df) // 10)])
    xticks = filtered_df['Time'][::max(1, len(filtered_df) // 10)]
    ax.set_xticklabels(xticks.dt.strftime('%Y-%m-%d'), rotation=45)
    label = 'Wind Speed at 100m (m/s)' if variable_name == 'windspeed_100m' else 'Power Output (normalized)'
    ax.set_xlabel('Time')
    ax.set_ylabel(label)
    ax.set_title(f'{label} for Site {site_index} ({start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d})')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return fig, ax


def evaluate_model(predictions, actuals):
    """
    Evaluate model predictions using MAE, MSE, and RMSE.
    """
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


def engineer_features(df):
    """
    Add useful features for wind power prediction.
    Assumes 'Time' is already a datetime column.
    """
    df = df.copy()
    df['hour'] = df['Time'].dt.hour
    df['month'] = df['Time'].dt.month
    df['dayofweek'] = df['Time'].dt.dayofweek
    df['Power_lag1'] = df['Power'].shift(1)
    df['Power_lag2'] = df['Power'].shift(2)
    df['Power_lag3'] = df['Power'].shift(3)

    if 'winddirection_100m' in df.columns:
        radians = np.deg2rad(df['winddirection_100m'])
        df['wind_dir_sin'] = np.sin(radians)
        df['wind_dir_cos'] = np.cos(radians)

    df.dropna(inplace=True)
    return df


class SVRModel:
    """Support Vector Regression model wrapper."""
    def __init__(self, train_df, target_col='Power'):
        self.target_col = target_col
        self.train_df = train_df
        self.features = [col for col in train_df.columns if col not in ['Time', self.target_col]]
        self.model = SVR(kernel='rbf', C=1, epsilon=0.03, max_iter=20000)
        self.scaler = StandardScaler()

    def train(self):
        X = self.train_df[self.features]
        y = self.train_df[self.target_col]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, test_df):
        X_test = test_df[self.features]
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


class GradientBoostingModel:
    """Gradient Boosting model wrapper."""
    def __init__(self, train_df, target_col='Power', n_estimators=100, learning_rate=0.1, max_depth=3):
        self.target_col = target_col
        self.train_df = train_df
        self.features = [col for col in train_df.columns if col not in ['Time', self.target_col]]
        self.model = GradientBoostingRegressor(n_estimators=n_estimators,
                                               learning_rate=learning_rate,
                                               max_depth=max_depth)
        self.scaler = StandardScaler()

    def train(self):
        X = self.train_df[self.features]
        y = self.train_df[self.target_col]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, test_df):
        X_test = test_df[self.features]
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


class RandomForestModel:
    """Random Forest model wrapper."""
    def __init__(self, train_df, target_col='Power'):
        self.target_col = target_col
        self.train_df = train_df
        self.features = [col for col in train_df.columns if col not in ['Time', self.target_col]]
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self):
        X = self.train_df[self.features]
        y = self.train_df[self.target_col]
        self.model.fit(X, y)

    def predict(self, test_df):
        X_test = test_df[self.features]
        return self.model.predict(X_test)


class FeedforwardNNModel:
    """Feedforward Neural Network model wrapper."""
    def __init__(self, train_df, target_col='Power', hidden_layer_sizes=(100, 50), max_iter=500):
        self.target_col = target_col
        self.train_df = train_df
        self.features = [col for col in train_df.columns if col not in ['Time', self.target_col]]
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
        self.scaler = StandardScaler()

    def train(self):
        X = self.train_df[self.features]
        y = self.train_df[self.target_col]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, test_df):
        X_test = test_df[self.features]
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


def get_input_file_path(site_index):
    """
    Returns the absolute path to the input CSV for the given site index.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(here, '..', '..'))
    return os.path.join(project_root, 'inputs', f'Location{site_index}.csv')


class PersistenceModel:
    """
    A persistence model for time series data that predicts the next value
    based on the previous value.
    """

    def __init__(self, df, target_column='Power'):
        """
        Initialize the PersistenceModel with a pandas DataFrame.

        Parameters:
        df (pd.DataFrame): The input data for the model.
        target_column (str): Column to predict.
        """
        self.df = df
        self.target = target_column
        self.preds = None
        self.true = None

    def predict(self):
        """
        Predict one hour ahead values using the persistence model.
        """
        self.preds = self.df[self.target].shift(1).iloc[1:]
        self.true = self.df[self.target].iloc[1:]
        return self.preds

    def evaluate(self):
        """
        Evaluate the model using MAE, MSE, and RMSE.

        Returns:
        tuple: MAE, MSE, RMSE
        """
        if self.preds is None or self.true is None:
            raise ValueError("You must call predict() before evaluate()")

        mae = np.mean(np.abs(self.preds - self.true))
        mse = np.mean((self.preds - self.true) ** 2)
        rmse = np.sqrt(mse)
        return mae, mse, rmse


def plot_power_predictions(site_index, timestamps, predictions, actual_values, ML_model, time_window=None):
    """
    Plot actual vs predicted power for a given time window.
    """
    results_df = pd.DataFrame({
        'Time': timestamps,
        'Actual Power': actual_values,
        'Predicted Power': predictions
    })

    if time_window:
        start_date, end_date = time_window
        results_df = results_df[
            (results_df['Time'] >= start_date) & (results_df['Time'] <= end_date)
        ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df['Time'], results_df['Actual Power'], label='Actual Power', color='blue')
    ax.plot(
        results_df['Time'],
        results_df['Predicted Power'],
        label=f'Predicted Power ({ML_model})',
        color='red', linestyle='--'
    )

    ax.set_title(f'Site {site_index}: Actual vs Predicted Power ({ML_model} Model)', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Normalized Power', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
    return fig, ax


def save_figure(fig, filename, subfolder='outputs'):
    """
    Save a matplotlib figure to a subfolder in the project root.

    Parameters:
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        Name of the file.
    subfolder : str
        Subfolder inside the project directory.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(project_root, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    fig.savefig(full_path)
    print(f"Figure saved to: {os.path.relpath(full_path, start=project_root)}")


class SiteSummary:
    """
    Class to load and summarize basic statistics from a wind site file.
    """

    def __init__(self, site_index):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        inputs_folder = os.path.join(project_root, 'inputs')
        file_name = f"Location{site_index}.csv"
        self.file_path = os.path.join(inputs_folder, file_name)
        self.site_index = site_index
        self.data = self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.file_path)
        df["Time"] = pd.to_datetime(df["Time"])
        return df

    def summarize(self):
        """
        Print and return basic summary statistics.
        """
        summary = {
            "Site Index": self.site_index,
            "Start Time": self.data["Time"].min(),
            "End Time": self.data["Time"].max(),
            "Total Hours": len(self.data),
            "Mean Wind Speed @100m (m/s)": round(self.data["windspeed_100m"].mean(), 2),
            "Mean Power Output (normalized)": round(self.data["Power"].mean(), 3),
            "Max Power Output": round(self.data["Power"].max(), 3),
            "Min Power Output": round(self.data["Power"].min(), 3),
            "Mean Temperature (C)": round(self.data["temperature_2m"].mean(), 2),
            "Mean Relative Humidity (%)": round(self.data["relativehumidity_2m"].mean(), 2)
        }

        print("\nSite Summary:")
        for key, val in summary.items():
            print(f"{key}: {val}")

        return summary


def train_test_split(df, test_size):
    """
    Split a DataFrame into training and testing sets.

    Parameters:
    df : pd.DataFrame
        Input dataset sorted by time.
    test_size : float
        Fraction of data to be used for testing.

    Returns:
    tuple : train_df, test_df
    """
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df


class GeneralWindTurbine:
    """
    Base wind turbine class with power curve modeling.
    """

    def __init__(self, rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, name=None):
        self.rotor_diameter = rotor_diameter
        self.hub_height = hub_height
        self.rated_power = rated_power
        self.v_in = v_in
        self.v_rated = v_rated
        self.v_out = v_out
        self.name = name

    def get_power(self, v):
        """
        Compute power output given wind speed v.
        """
        if v < self.v_in or v > self.v_out:
            return 0
        if self.v_in <= v < self.v_rated:
            return self.rated_power * (v / self.v_rated) ** 3
        if self.v_rated <= v <= self.v_out:
            return self.rated_power
        return 0


class WindTurbine(GeneralWindTurbine):
    """
    Wind turbine model using empirical power curve data.
    """

    def __init__(self, rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, power_curve_data, name=None):
        super().__init__(rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, name)
        self.power_curve_data = power_curve_data

    def get_power(self, v):
        """
        Interpolated power output based on wind speed.
        """
        wind_speeds = self.power_curve_data[:, 0]
        power_values = self.power_curve_data[:, 1]
        return float(np.interp(v, wind_speeds, power_values))
