import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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