import numpy as np

rng = np.random.default_rng(0)
class Model:
    def __init__(self):
        pass

    def predict(self):
        raise NotImplementedError("Model.predictは継承先でのみ使用できます.")

    def fit(self):
        raise NotImplementedError("Model.fitは継承先でのみ使用できます.")

    def _sample(self, X_train: np.ndarray, y_train: np.ndarray, batch_size: int) -> tuple[np.ndarray]:
        indices = rng.choice(len(X_train), batch_size)
        X_batch = X_train[indices]
        y_batch = y_train[indices]

        return X_batch, y_batch

    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        """0次元のnp.ndarray(int, float型含む), もしくは1次元のnp.ndarrayを2次元の縦ベクトルに変換する. 

        Args:
            X (np.ndarray): 2次元以下のnp.ndarray, int, float, list. 

        Raises:
            ValueError: 入力が3次元以上, もしくはnp.ndarray, int, float, list以外の型だった場合に発火.

        Returns:
            np.ndarray: 縦ベクトルとなったX. Xが2次元であった場合, そのままXが返される.
        """
        if isinstance(X, list):
            X = np.array(X)

        if isinstance(X, int) or isinstance(X, float):
            return np.atleast_2d(X)
        
        if X.ndim < 2:
            return np.atleast_2d(X).T
        
        elif X.ndim == 2:
            return X
        
        else:
            raise ValueError(f"モデルの入力は2次元以下のnp.ndarrayか, int, floatに限られます. type(X): {type(X)}")