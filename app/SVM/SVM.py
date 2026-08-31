import numpy as np

from model import Model

class SVMBinaryClassifer(Model):
    def __init__(self):
        pass

    def _get_K(self, X: np.ndarray, Y: np.ndarray, kernel_type: str, **kwargs):

        # K(n × n型行列)がメモリに負荷をかけるため, np.float64ではなくnp.float32にしておく. 
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)

        if kernel_type == "linear":

            return X @ Y.T
    
        elif kernel_type == "rbf":

            norm_train = np.sum(X ** 2, axis=1, keepdims=True)
            norm_test = np.sum(Y ** 2, axis=1, keepdims=True)

            gamma = kwargs.get("gamma")
            
            if gamma is None:
                gamma = 0.1

            dist_squared = norm_train - 2*(X @ Y.T) + norm_test.T
            K = np.exp(-gamma * dist_squared)

            return K
        
        else:
            raise ValueError(f"kernel_typr: {kernel_type}は存在しません. 'linear', 'rbf'が想定されます.")


    def _smo(self, y_train: np.ndarray, K: np.ndarray, C: float, max_iter: int =10000, seed: int =0, return_boundary_record: bool =False):
        
        if y_train.ndim == 2:
            y_train = y_train.flatten()

        N = len(y_train)

        tol = 1e-3 # aが境界(0 or C)に入っていると判定する際の許容誤差
        wss_epsilon = 1e-8 # a_iの変化量 < wss_epsilonのとき, working setはup, lowからそれぞれランダムにひとつずつ選ばれる
        convergence_epsilon = 1e-5 # upの最大値とlowの最小値が, convergence_epsilonより近くなったら, アルゴリズムが収束したと判定する. 

        a = np.zeros(N)

        # v = y_i - \sum_{i \in [n]}a_i y_i K_{ij}
        v = y_train.copy()

        rng = self._get_rng(seed=seed)

        # 最初のWorking Setはランダムな二変数. 
        s, t = rng.choice(N, 2, replace=False)

        up_max_record = []
        low_min_record = []

        for step in range(max_iter):

            # a_s(Platt. 1998 ではa_2に相当)の範囲を決定
            if y_train[s] != y_train [t]:
                L = max(0, a[s] - a[t])
                H = min(C, C + a[s] - a[t])
            else:
                L = max(0, a[s] + a[t] - C)
                H = min(C, a[s] + a[t])

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

            # Working Set Selection
            up_mask = ((a < C - tol) & (y_train == 1)) | ((a > tol) & (y_train == -1))
            low_mask = ((a < C - tol) & (y_train == -1)) | ((a > tol) & (y_train == 1))
    
            s_max = np.argmax(np.where(up_mask, v, -np.inf))
            t_min = np.argmin(np.where(low_mask, v, np.inf))

            up_max_record.append(v[s_max])
            low_min_record.append(v[t_min])

            if v[s_max] <= v[t_min] + convergence_epsilon:

                b = (v[s] + v[t]) / 2
                print("収束しました. ")
                return a, b

            # 前回からの更新量が極めて小さい場合, Working Set Selectionをランダムに行う.
            # つまり, upとlowからランダムに一つずつ選んで, それをWorking Setとする.
            if abs(delta_a_s) < wss_epsilon:
                s = rng.choice(np.arange(N)[up_mask])
                t = rng.choice(np.arange(N)[low_mask])
            else:
                s = s_max
                t = t_min

        else:
            b = (v[s] + v[t]) / 2
            print(f"max_iter={max_iter}回の試行では収束しませんでした. 暫定解が返されます. ")

        if return_boundary_record:
            return a, b, up_max_record, low_min_record
        else:
            return a, b

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, kernel_type: str ="rbf", C: float =1.0, max_iter=10000, return_boundary_record: bool =False, **kwargs):

        if len(X_train) != len(y_train):
            raise ValueError(f"X_train, y_trainの長さを揃えてださい. len(X_train): {len(X_train)}, len(y_train): {len(y_train)}")

        self.X_train = X_train
        self.y_train = y_train
        self.kernel_type = kernel_type
        self.K = self._get_K(X_train, X_train, kernel_type=kernel_type, **kwargs)

        result = self._smo(y_train=y_train, K=self.K, C=C, max_iter=max_iter, return_boundary_record=return_boundary_record)
        self.a, self.b = result[0], result[1]

        if return_boundary_record:
            return result[2], result[3]

    def predict(self, X_test: np.ndarray, return_score=False):
        a = np.reshape(self.a, (len(self.a), 1))
        y = np.reshape(self.y_train, (len(self.y_train), 1))
        K = self._get_K(self.X_train, X_test, kernel_type=self.kernel_type)
        score = np.sum(a * y * K, axis=0) + self.b

        if return_score:
            return score
        else:
            return np.sign(score)

class SVC(SVMBinaryClassifer):
    def __init__(self):
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, kernel_type: str ="rbf", C: float =1.0, max_iter: int =10000, **kwargs):
        self._fit_ovo(X_train, y_train, kernel_type=kernel_type, C=C, max_iter=max_iter, **kwargs)

    def predict(self, X_test: np.ndarray):
        return self._predict_ovo(X_test)

    def _fit_ovo(self, X_train: np.ndarray, y_train: np.ndarray, kernel_type: str ="rbf", C: float =1.0, max_iter=10000, **kwargs):
        class_set = np.unique(y_train).astype(int)
        class_n = len(class_set)
        self.models = []
        self.model_labels = {}

        for i in range(class_n):
            for j in range(class_n):
                if i < j:
                    class1_mask = y_train == class_set[i]
                    class2_mask = y_train == class_set[j]

                    X_class1 = X_train[class1_mask]
                    X_class2 = X_train[class2_mask]

                    X_train_target = np.concatenate([X_class1, X_class2], axis=0)
                    y_train_target = np.concatenate([np.ones(len(X_class1)), -np.ones(len(X_class2))], axis=0)

                    model = SVMBinaryClassifer()
                    model.fit(X_train_target, y_train_target, kernel_type=kernel_type, C=C, max_iter = max_iter, **kwargs)

                    # 1, -1というラベルが, 本来どのようなラベルであったか覚えさせる. 
                    self.model_labels[id(model)] = (class_set[i], class_set[j])
                    self.models.append(model)

    def _predict_ovo(self, X_test: np.ndarray):
        # データ数 × モデル数の, モデルの予測をまとめた行列を作る. 
        # i, j成分には, テストデータ X_test[i] に対する, self.models[j]の予測ラベルが入る.
        model_predictions = np.zeros((len(X_test), len(self.models)))

        for j, model in enumerate(self.models):

            prediction = model.predict(X_test)

            plabel, nlabel = self.model_labels[id(model)]

            prediction_with_label = np.where(prediction == 1, plabel, nlabel)

            model_predictions[:, j] = prediction_with_label

        # 多数決で集計
        predictions = np.zeros(len(X_test))

        for i in range(len(X_test)):

            labels, counts = np.unique(model_predictions[i], return_counts=True)
            predicton = labels[np.argmax(counts)]
            predictions[i] = predicton

        return predictions