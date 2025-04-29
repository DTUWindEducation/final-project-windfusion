import sys, os
import pandas as pd
import pytest

# Allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from finalproject.split_data import train_test_split

@pytest.fixture
def sample_df():
    """Create a sample dataframe with 10 hourly timestamps."""
    return pd.DataFrame({
        "Time": pd.date_range(start="2022-01-01", periods=10, freq="H"),
        "Power": range(10)
    })

def test_train_test_split_sizes(sample_df):
    """Test if the split returns the correct number of rows."""
    train, test = train_test_split(sample_df, test_size=0.3)
    expected_train_size = int(len(sample_df) * (1 - 0.3))
    expected_test_size = len(sample_df) - expected_train_size
    assert len(train) == expected_train_size
    assert len(test) == expected_test_size

def test_train_test_split_order(sample_df):
    """Test if the split maintains the time order."""
    train, test = train_test_split(sample_df, test_size=0.3)
    assert train["Time"].max() < test["Time"].min()
