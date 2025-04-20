import pandas as pd
import os
import matplotlib.pyplot as plt

def load_observations_data(file_name):
    """
    Load and parse observations dataset for one of the sites from a CSV file in the 'inputs' folder.
    """
    file_path = os.path.join("..", "inputs", file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, engine='python')  

    # Display basic information about the dataset
    print(f"Dataset loaded successfully from {file_path}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values detected:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values detected in the dataset.")

    # Display basic statistics
    basic_stats = df.describe(include='all').round(2)
    print("\nBasic statistics:")
    print(basic_stats)

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
