# src/finalproject/site_summary.py

import pandas as pd
import os

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
