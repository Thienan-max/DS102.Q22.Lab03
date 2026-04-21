import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.01, n_iterations=1000, C=1.0) -> None:
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.C = C
        self.w = None
        self.b = None

    def fit(self, X, y) -> None:
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1 
                
                if condition:
                    # TH1: Điểm an toàn (Hinge loss = 0)
                    self.w = self.w - self.lr * self.w 
                else:
                    # TH2: Điểm vi phạm margin (Hinge loss > 0)
                    self.w = self.w - self.lr * (self.w - self.C * y[idx] * x_i)
                    self.b = self.b + self.lr * self.C * y[idx]

    def predict(self, X) -> np.ndarray:
        y_pred = X @ self.w + self.b
        return np.where(y_pred >= 0, 1, -1)

    def evaluate(self, X, y) -> float:
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy