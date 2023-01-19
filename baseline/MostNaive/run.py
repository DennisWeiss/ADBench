import numpy as np


class MostNaive:
    def __init__(self, seed, model_name):
        self.center = None

    def fit(self, X_train, y_train):
        self.center = np.mean(X_train[y_train == 0], axis=0)
        return self

    def predict_score(self, x):
        if self.center is None:
            raise Exception("Model is not fitted")
        return np.linalg.norm(x - self.center, axis=1)