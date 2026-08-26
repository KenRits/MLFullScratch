from DecisionTree.decisionTree import DecisionTreeClassifer
from concurrent.futures import ProcessPoolExecutor
import exception
from model import Model

from matplotlib import pyplot as plt
import numpy as np

rng = np.random.default_rng(1)

class RandomForestClassifer(Model):
    def __init__(self):
        super().__init__()
        self.models = {}

    def _train_tree(self, model_idx, X_sample, y_sample, max_depth, bin_num, ccp_alpha, columns):
        model = DecisionTreeClassifer()
        model.idx = model_idx
        model.fit(X_sample, y_sample, max_depth=max_depth, bin_num=bin_num, ccp_alpha=ccp_alpha)
        return model, columns

    def fit(self, X_train, y_train, model_n: int =8, feature_n: int =None, sample_n: int =None, bootstrap=True, max_depth=10, ccp_alpha=None, bin_num=256):
        self.labels = np.unique(y_train)
        data_n, feature_dim = X_train.shape
        
        if feature_n is None:
            feature_n = min(feature_dim, 8)

        if sample_n is None:
            sample_n = min(data_n, 64)

        if feature_n > feature_dim:
            raise exception.InvalidDataShapeException(f"X_train.shape[1]: {X_train.shape[1]}, feature_n: {feature_n}")
        
        if not bootstrap and sample_n > data_n:
            raise exception.InvalidDataShapeException(f"X_train.shape[0]: {X_train.shape[0]}, sample_n: {sample_n}\nbootstrapを行わない場合, データ数以上のサンプルで決定木を構成することはできません. sample_nをデータ数以下にするか, bootstrapをTrueにしてください.")

        with ProcessPoolExecutor() as executer:

            futures = []

            for model_idx in range(model_n):

                columns = rng.choice(feature_dim, size=feature_n, replace=False)
                sample_indices = rng.choice(data_n, size=sample_n, replace=bootstrap)

                X_sample = X_train[np.ix_(sample_indices, columns)]
                y_sample = y_train[sample_indices]

                futures.append(executer.submit(self._train_tree, model_idx, X_sample, y_sample, max_depth, bin_num, ccp_alpha, columns))

            for future in futures:
                model, columns = future.result()
                # 各決定木の中で処理されている列のindexは, Random_Forestで処理されているindexとは異なるので, その対応付けが必要. 
                self.models[model.idx] = {"model": model, "column_idx_match": columns}
                print(f"model_{model.idx} finished.")

    def predict(self, X: np.ndarray):

        model_predictions = np.zeros((len(X), len(self.models)))
        
        for model_idx in self.models.keys():

            model = self.models[model_idx]["model"]
            X_translated = X[:, self.models[model_idx]["column_idx_match"]]
            model_prediction = model.predict(X_translated)
            model_predictions[:, model_idx] =  model_prediction

        prediction = np.zeros((1, len(X)))

        for i in range(len(model_predictions)):
            unique, counts = np.unique(model_predictions[i], return_counts=True)
            label = unique[np.argmax(counts)]
            prediction[:, i] = label

        return prediction