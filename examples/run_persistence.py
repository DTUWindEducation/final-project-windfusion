import numpy as np
import pandas as pd
import sys, os

# Define site index
site_index = 1

# Dynamically add src/ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from finalproject.persistence_model import PersistenceModel

# Build absolute path to the inputs folder and access input csv files
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
inputs_folder = os.path.join(project_root, 'inputs')
file_name = f"Location{site_index}.csv"
file_path = os.path.join(inputs_folder, file_name)

df = pd.read_csv(file_path)
df["Time"] = pd.to_datetime(df["Time"])

model = PersistenceModel(df)
model.predict()
mae, mse, rmse = model.evaluate()

print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
