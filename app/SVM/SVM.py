import numpy as np

from model import Model

class SVMBinaryClassifer(Model):
    def __init__(self):
        pass

    def _get_K(self, X_train: np.ndarray, kernel_type: str, **kwargs):

        if kernel_type == "linear":
            K = X_train @ X_train.T
            return K

        elif kernel_type == "rbf":

            gamma = kwargs.get("gamma")

            if gamma is None:
                gamma = 0.1

            self.gamma = gamma

            norm_squared = np.sum(X_train**2, axis=1, keepdims=True)
            dist_squared = norm_squared + norm_squared.T - 2 * X_train @ X_train.T
            K = np.exp(-gamma * dist_squared)

            return K

        else:
            raise ValueError(f"kernel_typr: {kernel_type}は存在しません. 'linear', 'rbf'が想定されます.")

    def _get_K_test(self, X_train: np.ndarray, X_test: np.ndarray, kernel_typre: str):
        if kernel_typre == "linear":
            return X_train @ X_test.T
    
        elif kernel_typre == "rbf":
            norm_train = np.sum(X_train ** 2, axis=1, keepdims=True)
            norm_test = np.sum(X_test ** 2, axis=1, keepdims=True)

            dist_squared = norm_train - 2*(X_train @ X_test.T) + norm_test.T
            K = np.exp(-self.gamma * dist_squared)

            return K


    def _smo(self, y_train: np.ndarray, K: np.ndarray, C: float, max_iter: int =10000, seed: int =0):

        if y_train.ndim == 2:
            y_train = y_train.flatten()

        N = len(y_train)

        tol = 1e-3

        a = np.zeros(N)

        # v = y_i - \sum_{i \in [n]}a_i y_i K_{ij}
        v = y_train.copy()

        rng = self._get_rng(seed=seed)

        s, t = rng.choice(N, 2, replace=False)

        for step in range(max_iter):

            # a_s(Platt. 1998 ではa_2に相当)の範囲を決定
            if y_train[s] != y_train [t]:
                L = max(0, a[s] - a[t])
                H = min(C, C + a[s] - a[t])
            else:
                L = max(0, a[s] + a[t] - C)
                H = min(C, a[s] + a[t])
            
            print(f"s: {s}")
            print(f"t: {t}")

            # a_s, a_tを更新
            eta = max(K[s, s] - 2*K[s, t] + K[t, t], 1e-8)

            a_s_new = a[s] + y_train[s] * (v[s] - v[t]) / eta
            a_s_new_clipped = np.clip(a_s_new, L, H)

            a_t_new = a[t] + y_train[s] * y_train[t] * (a[s] - a_s_new_clipped)

            delta_a_s = a_s_new_clipped - a[s]
            delta_a_t = a_t_new - a[t]

            a[s] = a_s_new_clipped
            a[t] = a_t_new

            # vを更新
            v = v - y_train[s] * delta_a_s * K[s, :] - y_train[t] * delta_a_t * K[t, :]

            # KKT条件の確認
            # L0_mask = np.logical_and(tol <= a, a <= C - tol)
            # L1_mask = np.logical_and(a < tol, y_train == 1)
            # L2_mask = np.logical_and(a < tol, y_train == -1)
            # L3_mask = np.logical_and(a > C - tol, y_train == 1)
            # L4_mask = np.logical_and(a > C - tol, y_train == -1)

            up_mask = ((a < C - tol) & (y_train == 1)) | ((a > tol) & (y_train == -1))
            low_mask = ((a < C - tol) & (y_train == -1)) | ((a > tol) & (y_train == 1))

            s = np.argmax(np.where(up_mask, v, -np.inf))
            t = np.argmin(np.where(low_mask, v, np.inf))

            if v[s] < v[t]:

                b = (v[s] + v[t]) / 2
                print("収束しました. ")
                return a, b
        
        b = (v[s] + v[t]) / 2

        print(f"max_iter: {max_iter}回の試行では収束しませんでした. 暫定解が返されます. ")

        return a, b

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, kernel_type: str ="rbf", C: float =1.0, max_iter=10000, **kwargs):

        if len(X_train) != len(y_train):
            raise ValueError(f"X_train, y_trainの長さを揃えてださい. len(X_train): {len(X_train)}, len(y_train): {len(y_train)}")

        self.X_train = X_train
        self.y_train = y_train

        self.kernel_type = kernel_type

        self.K = self._get_K(X_train, kernel_type=kernel_type, **kwargs)

        self.a, self.b = self._smo(y_train=y_train, K=self.K, C=C, max_iter=max_iter)

    def predict(self, X_test: np.ndarray):
        a = np.reshape(self.a, (len(self.a), 1))
        y = np.reshape(self.y_train, (len(self.y_train), 1))
        K = self._get_K_test(self.X_train, X_test, kernel_typre=self.kernel_type)
        prediction = np.sum(a * y * K, axis=0) + self.b

        return prediction
        







    