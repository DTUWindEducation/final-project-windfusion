import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
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
    Plot a time series of a selected variable for a given site within a specific period.
    
    Parameters:
    -----------
    variable_name : str
        The variable to plot ('windspeed_100m' or 'Power')
    site_index : int
        The site index (1, 2, 3, or 4)
    starting_time : str
        The starting time in format 'YYYY-MM-DD HH:MM:SS'
    ending_time : str
        The ending time in format 'YYYY-MM-DD HH:MM:SS'
    data_path : str, optional
        Path to the CSV file. If None, assumes a standard path format.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Validate inputs
    if variable_name not in ['windspeed_100m', 'Power']:
        raise ValueError("variable_name must be either 'windspeed_100m' or 'Power'")
    
    if site_index not in [1, 2, 3, 4]:
        raise ValueError("site_index must be 1, 2, 3, or 4")
    
    # Format the data path to point to the 'inputs' folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    inputs_folder = os.path.join(project_root, 'inputs')
    data_path = os.path.join(inputs_folder, f"Location{site_index}.csv")

    
    # Load the data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file {data_path} not found. Please provide the correct path.")
    
    # Convert Time column to datetime format
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
    else:
        raise ValueError("Time column not found in the data")
    
    # Check if variable exists in the dataframe
    if variable_name not in df.columns:
        raise ValueError(f"Variable '{variable_name}' not found in the data")
    
    # Convert start and end times to datetime
    start_dt = pd.to_datetime(starting_time)
    end_dt = pd.to_datetime(ending_time)
    
    # Filter data for the specified time period
    filtered_df = df[(df['Time'] >= start_dt) & (df['Time'] <= end_dt)]
    
    if filtered_df.empty:
        raise ValueError("No data found for the specified time period")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the data
    ax.plot(filtered_df['Time'], filtered_df[variable_name], 'b-', linewidth=1.5)
    
    # Format the x-axis to show dates nicely
    ax.set_xticks(filtered_df['Time'][::max(1, len(filtered_df)//10)])
    ax.set_xticklabels(filtered_df['Time'][::max(1, len(filtered_df)//10)].dt.strftime('%Y-%m-%d'), rotation=45)
    
    # Add labels and title
    variable_label = 'Wind Speed at 100m (m/s)' if variable_name == 'windspeed_100m' else 'Power Output (normalized)'
    ax.set_xlabel('Time')
    ax.set_ylabel(variable_label)
    ax.set_title(f'{variable_label} for Site {site_index} ({start_dt.strftime("%Y-%m-%d")} to {end_dt.strftime("%Y-%m-%d")})')
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()

    plt.show()
    
    return fig, ax

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

def engineer_features(df):
    """
    Add useful features for wind power prediction.
    Assumes 'Time' is already a datetime column.
    """

    df = df.copy()  # avoid modifying original

    # Time-based features
    df['hour'] = df['Time'].dt.hour
    df['month'] = df['Time'].dt.month
    df['dayofweek'] = df['Time'].dt.dayofweek

    # Lag features
    df['Power_lag1'] = df['Power'].shift(1)
    df['Power_lag2'] = df['Power'].shift(2)

    # Wind direction (convert degrees to radians, then sin/cos)
    if 'winddirection_100m' in df.columns:
        radians = np.deg2rad(df['winddirection_100m'])
        df['wind_dir_sin'] = np.sin(radians)
        df['wind_dir_cos'] = np.cos(radians)

    # Drop rows with NaNs introduced by shifts
    df.dropna(inplace=True)

    return df

class SVRModel:
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
    def __init__(self, train_df, target_col='Power', n_estimators=100, learning_rate=0.1, max_depth=3):
        self.target_col = target_col
        self.train_df = train_df
        self.features = [col for col in train_df.columns if col not in ['Time', self.target_col]]
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
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


class LagLinearModel:
    """
    A simple linear regression model using lagged values of the target as features.
    Serves as a proxy for long-term forecasting using past values.
    """
    def __init__(self, train_df, target_col='Power', lags=24):
        self.target_col = target_col
        self.lags = lags
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.train_df = self._prepare_lagged_data(train_df)

    def _prepare_lagged_data(self, df):
        df = df.copy()
        for lag in range(1, self.lags + 1):
            df[f'lag_{lag}'] = df[self.target_col].shift(lag)
        df.dropna(inplace=True)
        return df

    def train(self):
        X = self.train_df[[f'lag_{i}' for i in range(1, self.lags + 1)]]
        y = self.train_df[self.target_col]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, test_df):
        test_df = self._prepare_lagged_data(test_df)
        X_test = test_df[[f'lag_{i}' for i in range(1, self.lags + 1)]]
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    
    
    
class FeedforwardNNModel:
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


def plot_power_predictions(site_index, timestamps, predictions, actual_values, ML_model, time_window=None):

    # Create a DataFrame for plotting
    results_df = pd.DataFrame({
        'Time': timestamps,
        'Actual Power': actual_values,
        'Predicted Power': predictions
    })
    
    # Filter by time window if specified
    if time_window:
        start_date, end_date = time_window
        results_df = results_df[(results_df['Time'] >= start_date) & (results_df['Time'] <= end_date)]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df['Time'], results_df['Actual Power'], label='Actual Power', color='blue')
    ax.plot(results_df['Time'], results_df['Predicted Power'], label=f'Predicted Power ({ ML_model})', 
            color='red', linestyle='--')
    
    # Format the plot
    ax.set_title(f'Site {site_index}: Actual vs Predicted Power ({ ML_model} Model)', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Normalized Power', fontsize=12)
    
    # Format x-axis with dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()  # Rotate date labels
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    return fig, ax


class SiteSummary:
    def __init__(self, site_index):
        """
        Initializes the SiteSummary object and loads data from a consistent, absolute path.
        """
        # Build absolute path to the inputs folder
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
        Print basic statistics about the dataset.
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
        for k, v in summary.items():
            print(f"{k}: {v}")

        return summary
    

def train_test_split(df, test_size):
    """
    Split the dataframe into train and test sets based on time.
    Assumes df is sorted by time.
    """
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df


class GeneralWindTurbine:
    def __init__(self, rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, name=None):
        self.rotor_diameter = rotor_diameter
        self.hub_height = hub_height
        self.rated_power = rated_power
        self.v_in = v_in
        self.v_rated = v_rated
        self.v_out = v_out
        self.name = name

    def get_power(self, v):
        if v < self.v_in or v > self.v_out:
            return 0
        elif self.v_in <= v < self.v_rated:
            return self.rated_power * (v / self.v_rated) ** 3
        elif self.v_rated <= v <= self.v_out:
            return self.rated_power


class WindTurbine(GeneralWindTurbine):
    def __init__(self, rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, power_curve_data, name=None):
        super().__init__(rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, name)
        self.power_curve_data = power_curve_data

    def get_power(self, v):
        wind_speeds = self.power_curve_data[:, 0]
        power_values = self.power_curve_data[:, 1]
        # Use np.interp for linear interpolation
        return float(np.interp(v, wind_speeds, power_values))
