from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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
