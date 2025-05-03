import pytest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from matplotlib.figure import Figure
from types import SimpleNamespace
from finalproject import plot_feature_importance  

@pytest.fixture
def trained_model_and_features():
    X, y = make_regression(n_samples=100, n_features=15, noise=0.1, random_state=42)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return model, feature_names

def test_plot_feature_importance_runs(trained_model_and_features):
    model, feature_names = trained_model_and_features
    fig, ax = plot_feature_importance(model, feature_names, top_n=5, print_values=False)
    assert isinstance(fig, Figure)
    assert hasattr(ax, 'barh')

def test_plot_feature_importance_invalid_model():
    mock_model = SimpleNamespace()  # No feature_importances_
    feature_names = ['f1', 'f2', 'f3']
    with pytest.raises(ValueError, match="Model does not have feature_importances_ attribute."):
        plot_feature_importance(mock_model, feature_names)

def test_plot_feature_importance_prints(capsys, trained_model_and_features):
    model, feature_names = trained_model_and_features
    plot_feature_importance(model, feature_names, top_n=3, print_values=True)
    captured = capsys.readouterr()
    assert "Top Feature Importances:" in captured.out
