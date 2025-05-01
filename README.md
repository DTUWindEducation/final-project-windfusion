# Wind Power Forecasting Package (WindFusion)

## Overview

This package provides tools for analyzing wind farm data and developing machine learning models to predict wind power output. It enables data processing, feature engineering, model training, and visualization of wind power time series data, helping wind farm operators and energy analysts optimize their operations and forecasting capabilities.

## Project Information

- **Name**: finalproject
- **Description**: Team WindFusion Final Project Wind Power Forecasting
- **Version**: 0.1.0
- **Authors**:
  - T. Labeikis (s243092@student.dtu.dk)
  - G. Georgiadis (s243196@student.dtu.dk)
  - A.K. Sfakianoudis (s242856@student.dtu.dk)
- **Repository**: [https://github.com/DTUWindEducation/final-project-windfusion.git](https://github.com/DTUWindEducation/final-project-windfusion.git)

## Dependencies

- pandas
- numpy
- matplotlib
- scikit-learn

## Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/DTUWindEducation/final-project-windfusion.git
   cd final-project-windfusion
   ```

2. Install the required dependencies:
   ```bash
   pip install external-package
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

## Package Architecture

The package follows a modular architecture with clear separation of concerns:

```
final-project-windfusion/
├── src/
│   └── finalproject/
│       └── __init__.py     
├── inputs/
│   ├── Location1.csv
│   ├── Location2.csv
│   ├── Location3.csv
│   └── Location4.csv
├── examples/
│   └── main.py
├── outputs/
│   └── [generated plots]
├── tests/
│   └── test_finalproject.py
├── pyproject.toml
├── LICENSE
├── Collaboration.md
└── README.md
```

### Architecture Diagram (Logical View)

```
┌───────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│                   │     │                  │     │                   │
│   Data Loading    │────▶│  Data Processing │────▶│  Model Training   │
│   & Preparation   │     │  & Engineering   │     │  & Evaluation     │
│                   │     │                  │     │                   │
└───────────────────┘     └──────────────────┘     └─────────┬─────────┘
                                                             │
┌───────────────────┐                           ┌─────────────▼─────────┐
│                   │                           │                       │
│    Persistence    │◀──────────────────────────│    ML Model           │
│    Benchmarking   │                           │    Prediction         │
│                   │                           │                       │
└─────────┬─────────┘                           └─────────────┬─────────┘
          │                                                   │
          │           ┌───────────────────────────┐           │
          └──────────▶│                           │◀──────────┘
                      │     Visualization &       │
                      │     Performance           │
                      │     Evaluation            │
                      │                           │
                      └───────────────────────────┘
```

### Architecture Diagram (Development View)
```
                    +-----------------------------+
                    |      examples/main.py|
                    +-------------+---------------+
                                  |
                                  v
+-------------------------------------------------------------+
|                          src/finalproject/__init__.py       |
| Main controller module: imports and coordinates models,     |
| utilities, feature engineering, evaluation, and plotting.   |
+-------------------------------------------------------------+
   |         |           |               |              |
   |         |           |               |              |
   v         v           v               v              v
+--------+ +--------------------+ +----------------+ +------------------+ +-------------------+
| utils  | | feature_engineering| |    model       | |   evaluation     | |     plotting      |
|        | |                    | |                | |                  | |                   |
+--------+ +--------------------+ +----------------+ +------------------+ +-------------------+
   |           |                        |                   |                    |
   |           |                        |                   |                    |
   v           v                        v                   v                    v
[Load CSVs] [Add datetime & lags]   [Train/test ML]     [MAE/MSE/RMSE]     [TS & pred plots]
[Split data][Handle missing vals]  [Predict values]    [Persistence model] [Matplotlib-based]
```

### Architecture Diagram (Process View)
```
+----------------------+
|   Start (main.py)    |
+----------+-----------+
           |
           v
+-----------------------------+
| get_input_file_path(site)  |
|  → Load CSV data           |
+-----------------------------+
           |
           v
+-----------------------------+
|  engineer_features(df)     |
|  → Create lags, wind power |
+-----------------------------+
           |
           v
+-----------------------------+
| train_test_split(df)       |
|  → 80/20 split by time     |
+-----------------------------+
           |
           v
+-----------------------------+
|  Select ML Model           |
|  e.g., SVRModel(train_df)  |
+-----------------------------+
           |
           v
+-----------------------------+
|  model.train()             |
|  → Fit on train_df         |
+-----------------------------+
           |
           v
+-----------------------------+
|  model.predict(test_df)    |
|  → Generate predictions    |
+-----------------------------+
           |
           v
+-----------------------------+
|  evaluate_model()          |
|  → MAE, MSE, RMSE          |
+-----------------------------+
           |
           v
+-----------------------------+
|  plot_power_predictions()  |
|  → Save to /outputs        |
+-----------------------------+
           |
           v
+-----------------------------+
|  plot_timeseries()         |
|  → Save to /outputs        |
+-----------------------------+
           |
           v
+-----------------------------+
| PersistenceModel(test_df)  |
|  → Predict and evaluate    |
+-----------------------------+
           |
           v
+----------------------+
|   End of Program     |
+----------------------+
```

## Package Components

### Classes

#### Data Handling and Processing

1. **SiteSummary** (`src/finalproject/__init__.py`)
   - Provides statistical summaries of data for a specific site.
   - Methods:
     - `__init__(site_index)`: Initializes with site index and loads data
     - `_load_data()`: Loads CSV data for the specified site
     - `summarize()`: Generates and prints statistical summaries

#### Machine Learning Models

2. **SVRModel** (`src/finalproject/__init__.py`)
   - Support Vector Regression model for wind power prediction
   - Methods:
     - `__init__(train_df, target_col='Power')`: Initializes model parameters
     - `train()`: Trains the SVR model on the provided data
     - `predict(test_df)`: Makes predictions on test data

3. **GradientBoostingModel** (`src/finalproject/__init__.py`)
   - Gradient Boosting Regression model for wind power prediction
   - Methods:
     - `__init__(train_df, target_col='Power', n_estimators=100, learning_rate=0.1, max_depth=3)`
     - `train()`: Trains the GB model on the provided data
     - `predict(test_df)`: Makes predictions on test data

4. **LagLinearModel** (`src/finalproject/__init__.py`)
   - Linear regression model using lagged values for time series forecasting
   - Methods:
     - `__init__(train_df, target_col='Power', lags=24)`
     - `_prepare_lagged_data(df)`: Creates lag features for the model
     - `train()`: Trains the linear model on the lagged data
     - `predict(test_df)`: Makes predictions on test data

5. **FeedforwardNNModel** (`src/finalproject/__init__.py`)
   - Neural network model for wind power prediction
   - Methods:
     - `__init__(train_df, target_col='Power', hidden_layer_sizes=(100, 50), max_iter=500)`
     - `train()`: Trains the neural network model
     - `predict(test_df)`: Makes predictions on test data

6. **PersistenceModel** (`src/finalproject/__init__.py`)
   - Simple baseline model that uses the previous value as a prediction
   - Methods:
     - `__init__(df, target_column='Power')`
     - `predict()`: Makes predictions based on persistence
     - `evaluate()`: Calculates performance metrics


### Utility Functions

- `load_observations_data(file_path)`: Loads and parses observation datasets
- `plot_timeseries(variable_name, site_index, starting_time, ending_time)`: Visualizes time series data
- `evaluate_model(predictions, actuals)`: Calculates performance metrics
- `engineer_features(df)`: Creates derived features for model training
- `train_test_split(df, test_size)`: Splits data into training and testing sets
- `get_input_file_path(site_index)`: Gets the absolute path to input files
- `plot_power_predictions(site_index, timestamps, predictions, actual_values, ML_model, time_window)`: Visualizes model predictions against actual values

## Example Usage

The package includes an example script (`examples/main.py`) that demonstrates how to:
1. Load data for a specific wind farm location
2. Process and engineer features
3. Train and evaluate models
4. Generate visualizations of results

## Peer Review

The package has undergone the following peer review processes: None

## License

This project is licensed under the MIT License - see the LICENSE file for details.