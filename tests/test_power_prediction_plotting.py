import unittest
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from finalproject import plot_power_predictions

class TestPlotPowerPredictions(unittest.TestCase):
    def setUp(self):
        self.site_index = 1
        self.model_name = "TestModel"
        self.timestamps = pd.date_range(start="2023-01-01", periods=10, freq='D')
        self.actual_values = [i * 0.1 for i in range(10)]
        self.predictions = [i * 0.1 + 0.05 for i in range(10)]

    def test_plot_output(self):
        fig, ax = plot_power_predictions(
            self.site_index,
            self.timestamps,
            self.predictions,
            self.actual_values,
            self.model_name
        )

        # Check that the returned object types are correct
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

        # Check that both lines are plotted
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)

        # Check labels of lines
        line_labels = [line.get_label() for line in lines]
        self.assertIn('Actual Power', line_labels)
        self.assertIn(f'Predicted Power ({self.model_name})', line_labels)

        # Check axis labels
        self.assertEqual(ax.get_xlabel(), 'Time')
        self.assertEqual(ax.get_ylabel(), 'Normalized Power')

        # Check title
        expected_title = f'Site {self.site_index}: Actual vs Predicted Power ({self.model_name} Model)'
        self.assertEqual(ax.get_title(), expected_title)

    def test_time_window_filtering(self):
        # Use a narrower window
        time_window = (self.timestamps[2], self.timestamps[5])
        fig, ax = plot_power_predictions(
            self.site_index,
            self.timestamps,
            self.predictions,
            self.actual_values,
            self.model_name,
            time_window=time_window
        )

        times_plotted = ax.get_lines()[0].get_xdata()
        for t in times_plotted:
            self.assertTrue(time_window[0] <= pd.to_datetime(t) <= time_window[1])