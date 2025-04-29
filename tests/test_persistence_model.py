# tests/test_persistence_model.py

import os, sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from finalproject.persistence_model import PersistenceModel
from finalproject.evaluation import evaluate_model

def test_persistence_model_prediction_and_evaluation():
    # Create a dummy DataFrame
    data = {
        'Time': pd.date_range(start='2021-01-01 00:00:00', periods=5, freq='h'),
        'Power': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    df = pd.DataFrame(data)

    model = PersistenceModel(df)
    preds = model.predict()

    # Check prediction shape (should match except for first NaN)
    assert len(preds) == 4
    assert abs(preds.iloc[0] - 0.1) < 1e-6

    # Evaluate (ignoring the first row, as done in your main code)
    mae, mse, rmse = evaluate_model(preds, df['Power'].iloc[1:])
    assert mae >= 0
    assert mse >= 0
    assert rmse >= 0
