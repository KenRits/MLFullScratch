import numpy as np
from exception import InvalidDataShapeException, FunctionUndefinedException

class Model:
    def __init__(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise InvalidDataShapeException(f"X.shape[0]: {X.shape[0]}, y.shape[0]: {y.shape[0]}")
                
        self.X = X
        self.y = y
        self.labels = np.unique(y)
        self.n = X.shape[0]
        self.d = X.shape[1]

    def fit(self):
        raise FunctionUndefinedException("Model.fit Modelクラスを継承したクラスを使用してください." )

    def predict(self):
        raise FunctionUndefinedException("Model.predict Modelクラスを継承したクラスを使用してください.")

    def predict_all(self):
        raise FunctionUndefinedException("Model.predict_all Modelクラスを継承したクラスを使用してください.")