# tests/test_feature_engineer.py

import os, sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from finalproject import engineer_features

def test_engineer_features():
    # Create a simple dummy dataframe
    data = {
        'Time': pd.date_range(start='2021-01-01 00:00:00', periods=5, freq='h'),
        'Power': [0.1, 0.2, 0.3, 0.4, 0.5],
        'winddirection_100m': [0, 90, 180, 270, 360]
    }
    df = pd.DataFrame(data)

    # Apply feature engineering
    result = engineer_features(df)

    # Check that new columns exist (updated to include Power_lag3)
    expected_columns = ['hour', 'month', 'dayofweek', 'Power_lag1', 'Power_lag2', 'Power_lag3', 'wind_dir_sin', 'wind_dir_cos']
    for col in expected_columns:
        assert col in result.columns, f"{col} not created"

    # Updated assertion: 5 original rows - 3 lags = 2 remaining rows
    assert len(result) == 2, f"Expected 2 rows after lag features, got {len(result)}"