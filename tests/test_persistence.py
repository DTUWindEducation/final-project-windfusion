import sys, os
import pandas as pd
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from finalproject import PersistenceModel

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Time": pd.date_range(start="2022-01-01", periods=10, freq="h"),
        "Power": np.linspace(0.1, 1.0, 10)
    })

def test_persistence_predict_length(sample_df):
    model = PersistenceModel(sample_df)
    preds = model.predict()
    assert len(preds) == len(sample_df) - 1

def test_persistence_evaluate_returns_floats(sample_df):
    model = PersistenceModel(sample_df)
    model.predict()
    mae, mse, rmse = model.evaluate()
    assert all(isinstance(x, float) for x in [mae, mse, rmse])