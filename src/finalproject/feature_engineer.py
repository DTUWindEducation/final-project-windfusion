import pandas as pd
import numpy as np

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
