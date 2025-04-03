import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import WindFusion  # Importing the WindFusion package

DATA_DIR = Path('./inputs')

def test_load_observations_data():
    """Test if data loads correctly and is parsed"""
    file_name = DATA_DIR / 'Location1.csv'  # Ensure this file exists in 'inputs' folder
    df, basic_stats = WindFusion.load_observations_data(file_name)  # Using WindFusion to load data
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'Time' in df.columns
    assert 'windspeed_100m' in df.columns
    assert 'Power' in df.columns
    assert 'windspeed_100m' in basic_stats.columns
    assert 'Power' in basic_stats.columns
    assert 'mean' in basic_stats.index
    assert 'std' in basic_stats.index

def test_plot_timeseries():
    """Test if the plot function generates a figure."""
    variable_name = "windspeed_100m"
    site_index = 1
    starting_time = "2017-05-01 00:00:00"
    ending_time = "2020-05-01 23:59:59"
    fig, ax = WindFusion.plot_timeseries(variable_name, site_index, starting_time, ending_time)  
    assert isinstance(fig, plt.Figure)
    assert ax is not None 
    assert len(ax.lines) > 0
   
