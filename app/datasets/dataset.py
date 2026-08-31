import numpy as np
from matplotlib import pyplot as plt

class Dataset:
    def __init__(self, name: str, description: str, X: np.ndarray, y: np.ndarray):
        self.name = name
        self.description = description
        self.X = X
        self.y = y

    def plot(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots()

        if self.X.shape[1] == 2:

            for label in self.labels:
                X_in_class = self.X[self.y == label]
                ax.scatter(X_in_class.T[0], X_in_class.T[1], label=label)
        
        elif self.X.shape[1] == 1 and self.y.shape[1] == 1:
            ax.scatter(self.X, self.y, alpha=0.8, label="Training Data")
        
        ax.grid()
        ax.set_title(f"{self.name} Dataset")
        ax.legend()

        return ax

    def meshgrid(self, n=100):
        d = self.X.shape[1]

        # メモリ管理の観点から, 多すぎる点を扱わないようにする.
        if n ** d > 1e+6:
            raise ValueError(f"1,000,000点を超える点群を扱うことは推奨されません. n ** self.X.shape[1]: {n ** d}")

        xxs = np.meshgrid(*(np.linspace(*self.domain[i], num=n) for i in range(d)))

        return xxs

    
    @property
    def labels(self):
        return np.unique(self.y)

    @property
    def class_n(self):
        return len(self.labels)

    @property
    def y_onehot(self):
        return np.eye(self.class_n)[self.y]

    @property
    def domain(self):
        return [(np.min(self.X.T[i]), np.max(self.X.T[i])) for i in range(self.X.shape[1])]

    def __repr__(self):
        f"# {self.name}\n {self.description}"
        return 

class ToyDatasetGenerator:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed=seed)

    def get_spiral(self, n: int, class_n: int, sigma: float =0.1, a: float =1.0):

        data_nums = self._get_data_num_per_class(n, class_n)

        X = np.zeros((n, 2), dtype=np.float32)
        y = np.zeros(n, dtype=int)

        for label in range(class_n):

            data_num = data_nums[label]
            index_start = np.sum(data_nums[:label])

            theta_bias = 2*np.pi * (label / class_n)

            theta = np.array([np.linspace(0, 3.5, data_num)]).T
            r = a * theta

            x1 = r * np.cos(theta + theta_bias) + self.rng.normal(0, sigma, theta.shape)
            x2 = r * np.sin(theta + theta_bias) + self.rng.normal(0, sigma, theta.shape)

            X[index_start:index_start+data_num, :] = np.concatenate([x1, x2], axis=1)
            y[index_start:index_start+data_num] = label * np.ones(data_num)

        X, y = self._shuffle(X, y)

        description = "非線形分類器用の2次元のデータセット. x_1, x_2を平面にプロットすると螺旋型の模様が現れる."

        return Dataset(name="Spiral", description=description, X=X, y=y)

    def get_sin_curve(self, n: int, sigma=0.1):

        X = np.array([np.linspace(-3, 3, n)]).T
        y = np.sin(X) + self.rng.normal(0, sigma, X.shape)

        X, y = self._shuffle(X, y)

        description = "非線形回帰モデル用の, R^1->R^1のデータセット. yはsin(X) (X \\in [-3, 3)) にしたがう."

        return Dataset(name="Sin curve", description=description, X=X, y=y)

    def get_square(self, n: int, sigma=0.5):
        X = np.array([np.linspace(-3, 3, n)]).T
        y = X**2 + self.rng.normal(0, sigma, X.shape)

        X, y = self._shuffle(X, y)

        description = "非線形回帰モデル用の, R^1->R^1のデータセット. yはX**2 (X \\in [-3, 3)) にしたがう."

        return Dataset(name="Square", description=description, X=X, y=y)

    @property
    def spiral(self):
        return self.get_spiral(n=100, class_n=3)

    @property
    def sin_curve(self):
        return self.get_sin_curve(n=100)

    @property
    def square(self):
        return self.get_square(n=100)

    def _get_data_num_per_class(self, n: int, class_n: int):

        if n < class_n:
            raise ValueError(f"データの数'n'はクラスの数'class_n'より大きく設定してください. n: {n}, class_n: {class_n}")
        
        data_nums = (n // class_n * np.ones(class_n)).astype(int)
        data_nums[:n % class_n] += 1
        return data_nums

    def _shuffle(self, X: np.ndarray, y: np.ndarray):
        indices = np.arange(len(X))
        self.rng.shuffle(indices)
        return X[indices], y[indices]

