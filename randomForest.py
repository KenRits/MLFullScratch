from decisionTree import DecisionTree
import preprocessing
from concurrent.futures import ProcessPoolExecutor
import exception
from model import Model

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)

class RandomForest(Model):
    def __init__(self, X, y):
        super().__init__(X, y)
        self.models = {}

    def _train_tree(self, model_idx, X_sample, y_sample, max_depth, bin_num, ccp_alpha, columns):
        model = DecisionTree(X_sample, y_sample)
        model.idx = model_idx
        model.fit(max_depth, bin_num, ccp_alpha)
        return model, columns

    def fit(self, model_n: int, feature_n: int, sample_n: int, bootstrap=True, max_depth=10, ccp_alpha=None, bin_num=256):

        if feature_n > self.d:
            raise exception.InvalidDataShapeException(f"X.shape[1]: {X.shape[1]}, feature_n: {feature_n}")
        
        if not bootstrap and sample_n > self.n:
            raise exception.InvalidDataShapeException(f"X.shape[0]: {X.shape[0]}, sample_n: {sample_n}\nbootstrapを行わない場合, データ数以上のサンプルで決定木を構成することはできません. sample_nをデータ数以下にするか, bootstrapをTrueにしてください.")

        with ProcessPoolExecutor() as executer:

            futures = []

            for model_idx in range(model_n):

                columns = rng.choice(self.d, size=feature_n, replace=False)
                sample_indices = rng.choice(self.n, size=sample_n, replace=bootstrap)

                X_sample = self.X[np.ix_(sample_indices, columns)]
                y_sample = self.y[sample_indices]

                futures.append(executer.submit(self._train_tree, model_idx, X_sample, y_sample, max_depth, bin_num, ccp_alpha, columns))

            for future in futures:
                model, columns = future.result()
                # 各決定木の中で処理されている列のindexは, Random_Forestで処理されているindexとは異なるので, その対応付けが必要. 
                self.models[model.idx] = {"model": model, "column_idx_match": columns}
                print(f"model_{model.idx} finished.")

    def predict(self, x):
        model_predictions = np.zeros(len(self.models))

        for model_idx in self.models.keys():
            model = self.models[model_idx]["model"]
            x_translated = x[self.models[model_idx]["column_idx_match"]]
            model_prediction = model.predict(x_translated)
            model_predictions[model_idx] =  model_prediction
        
        labels, counts = np.unique(model_predictions, return_counts=True)
        mode = labels[np.argmax(counts)]

        return mode

    def predict_all(self, X):
        result = np.zeros(len(X))
        for i, x in enumerate(X):
            result[i] = self.predict(x)
        return result

if __name__ == "__main__":
    df = pd.read_csv("winequality-red.csv")
    y = (df["quality"] >= 6.5).astype(int).values
    X = df.drop("quality", axis=1).values

    model = RandomForest(X, y)
    model.fit(model_n=50, feature_n=10, sample_n=500, bootstrap=True, max_depth=20, ccp_alpha=1.0)
    predicted = np.zeros(len(X))

    for i in range(len(X)):
        predicted_label = model.predict(X[i])
        predicted[i] = predicted_label

    print(1 - sum(np.logical_xor(predicted, y)) / len(y))
    