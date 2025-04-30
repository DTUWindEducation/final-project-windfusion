
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor

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
    
    

class XGBModel:
    def __init__(self, train_df, target_col='Power', n_estimators=100, max_depth=5, learning_rate=0.1):
        self.target_col = target_col
        self.train_df = train_df
        self.features = [col for col in train_df.columns if col not in ['Time', self.target_col]]
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror'
        )
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
  
    
   
class LSTMModel:
    def __init__(self, train_df, target_col='Power', time_steps=24, units=50, epochs=50, batch_size=32):
        self.target_col = target_col
        self.train_df = train_df
        self.time_steps = time_steps
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size

        self.features = [col for col in train_df.columns if col not in ['Time', self.target_col]]
        self.scaler = StandardScaler()
        self.model = None

    def _create_sequences(self, df):
        X, y = [], []
        for i in range(self.time_steps, len(df)):
            X.append(df[i - self.time_steps:i, :])
            y.append(self.y_data[i])
        return np.array(X), np.array(y)

    def train(self):
        X_raw = self.train_df[self.features].values
        self.y_data = self.train_df[self.target_col].values

        X_scaled = self.scaler.fit_transform(X_raw)
        X_seq, y_seq = self._create_sequences(X_scaled)

        self.model = Sequential()
        self.model.add(LSTM(self.units, input_shape=(X_seq.shape[1], X_seq.shape[2])))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

        self.model.fit(X_seq, y_seq, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, test_df):
        X_raw = test_df[self.features].values
        X_scaled = self.scaler.transform(X_raw)

        # Create sequences from test data
        X_seq = []
        for i in range(self.time_steps, len(X_scaled)):
            X_seq.append(X_scaled[i - self.time_steps:i])
        X_seq = np.array(X_seq)

        return self.model.predict(X_seq).flatten()
