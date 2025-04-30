import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

class SVRModel:
    def __init__(self, train_df, target_col='Power'):
        self.target_col = target_col
        self.train_df = train_df
        self.features = [col for col in train_df.columns if col not in ['Time', self.target_col]]
        self.model = SVR(kernel='rbf', C=1, epsilon=0.03, max_iter=20000)
        self.scaler = StandardScaler()

    def train(self):
        X = self.train_df[self.features]
        y = self.train_df[self.target_col]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, test_df):
        X_test = test_df[self.features]
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


class GradientBoostingModel:
    def __init__(self, train_df, target_col='Power', n_estimators=100, learning_rate=0.1, max_depth=3):
        self.target_col = target_col
        self.train_df = train_df
        self.features = [col for col in train_df.columns if col not in ['Time', self.target_col]]
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        self.scaler = StandardScaler()

    def train(self):
        X = self.train_df[self.features]
        y = self.train_df[self.target_col]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, test_df):
        X_test = test_df[self.features]
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


class LagLinearModel:
    """
    A simple linear regression model using lagged values of the target as features.
    Serves as a proxy for long-term forecasting using past values.
    """
    def __init__(self, train_df, target_col='Power', lags=24):
        self.target_col = target_col
        self.lags = lags
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.train_df = self._prepare_lagged_data(train_df)

    def _prepare_lagged_data(self, df):
        df = df.copy()
        for lag in range(1, self.lags + 1):
            df[f'lag_{lag}'] = df[self.target_col].shift(lag)
        df.dropna(inplace=True)
        return df

    def train(self):
        X = self.train_df[[f'lag_{i}' for i in range(1, self.lags + 1)]]
        y = self.train_df[self.target_col]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, test_df):
        test_df = self._prepare_lagged_data(test_df)
        X_test = test_df[[f'lag_{i}' for i in range(1, self.lags + 1)]]
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    
    
    
class FeedforwardNNModel:
    def __init__(self, train_df, target_col='Power', hidden_layer_sizes=(100, 50), max_iter=500):
        self.target_col = target_col
        self.train_df = train_df
        self.features = [col for col in train_df.columns if col not in ['Time', self.target_col]]
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
        self.scaler = StandardScaler()

    def train(self):
        X = self.train_df[self.features]
        y = self.train_df[self.target_col]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, test_df):
        X_test = test_df[self.features]
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
