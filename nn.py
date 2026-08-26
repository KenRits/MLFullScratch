import function as f
import optimizer as opt
from animator import ClassificationPainter, RegressionPainter

Variable = f.Variable
Parameter = f.Parameter

import numpy as np
from matplotlib import pyplot as plt


rng = np.random.default_rng(0)

class Layer:

    def __init__(self):
        self.params = []
    
    def forward(self):
        raise NotImplementedError(f"{self.__class__.__name__}.forwardは継承先でのみ使用できます.")

    def get_params(self):
        return self.params

    def __setattr__(self, name, value):
        # Parameterインスタンスがクラス変数として代入されると, それをself.paramsに追加する.
        super().__setattr__(name, value)
        if isinstance(value, Parameter):
            self.params.append(value)

class AffineLayer(Layer):

    def __init__(self, inp_dim: int, out_dim: int):
        super().__init__()
        self.W = Parameter(rng.normal(0, 1, (out_dim, inp_dim)))
        self.b = Parameter(rng.normal(0, 1, (out_dim, 1)))
    
    def forward(self, X: Variable):
        return self.W @ X + self.b

class BatchNormalizationLayer(Layer):
    def __init__(self, dim):
        super().__init__()
        self.gamma = Parameter(rng.normal(0, 0.1, (dim, 1)))
        self.beta = Parameter(rng.normal(0, 0.1, (dim, 1)))

    def forward(self, X: Variable, epsilon=1e-8):
        batch_size = X.shape[1]
        s = f.Sum()
        mean = s(X, axis=1) / batch_size

        s2 = f.Sum()
        std = s2((X - mean)**2 / batch_size, axis=1)**0.5

        r = self.gamma * (X - mean) / (std + epsilon) + self.beta

        return r

class FunctionLayer(Layer):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *inps):
        return self.func(*inps)

class ReLuLayer(FunctionLayer):
    def __init__(self):
        super().__init__(f.ReLu())

class Model:
    def __init__(self):
        pass

    def predict(self):
        raise NotImplementedError("Model.predictは継承先でのみ使用できます.")

    def fit(self):
        raise NotImplementedError("Model.fitは継承先でのみ使用できます.")

    def _sample(self, X_train: np.ndarray, y_train: np.ndarray, batch_size: int) -> tuple[Variable]:
        indices = rng.choice(len(X_train), batch_size)
        X_batch = Variable(X_train[indices].T)
        y_batch = Variable(y_train[indices].T)

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

class NeuralNetwork(Model):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, inp: Variable):
        x = inp
        for layer in self.layers:
            x = layer.forward(x)
        return x

    @property
    def params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.params)
        return params

class StandardMLP(NeuralNetwork):
    def __init__(self, unit_nums: list[int], task: str):

        self.task = task
        
        layers = []

        for l in range(len(unit_nums)-1):

            affine = AffineLayer(unit_nums[l], unit_nums[l+1])
            layers.append(affine)

            act_func_layer = FunctionLayer(f.SiLU())
            layers.append(act_func_layer)

        super().__init__(layers)

        # 出力層の活性化関数と損失関数を同時に.
        if task == "r": 
            # 回帰
            self.loss_layer = FunctionLayer(f.MSELoss())
        
        elif task == "c":
            # 多クラス分類
            self.loss_layer = FunctionLayer(f.CrossEntropyLoss())

        else:
            raise ValueError("引数taskには'r'(回帰)もしくは'c'(多クラス分類)が想定されています.")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, batch_size: int =None, alpha: float=0.01, max_iter: int =1000, painter=None):

        if len(X_train) != len(y_train):
            raise ValueError(f"X_train, y_trainの長さを揃えてださい. len(X_train): {len(X_train)}, len(y_train): {len(y_train)}")

        if batch_size is None:
            batch_size = min(10, len(X_train))

        optimizer = opt.Adam(self.params)

        loss_record = []

        for step in range(max_iter):

            X_batch, y_batch = self._sample(X_train, y_train, batch_size)

            digit = self.forward(X_batch)

            loss = self.loss_layer.forward(digit, y_batch)

            loss_record.append(loss.data[0][0])
            loss.backward()

            optimizer.step(alpha=alpha)

            # ア二メーション用の画像を作成.
            if painter is not None and step % 100 == 0:
                painter.plot_prediction(self, X_train, y_train)

        if painter is not None:
            painter.animate()

        return loss_record

    def predict(self, X: np.ndarray):
        X = self._to_2d(X)
        X_var = Variable(X.T)
        out = self.forward(X_var)

        if self.task == "r":
            prediction = out.data.T

        elif self.task == "c":
            prediction = self._to_2d(np.argmax(out.data.T, axis=1))
        
        else:
            raise ValueError(f"引数taskには'r'(回帰)もしくは'c'(多クラス分類)が想定されています. self.task: {self.task}")

        return prediction






if __name__ == "__main__":

    N = 1000
    lim = (-5, 5)

    # データ作成(回帰)
    # X_train = np.array([np.linspace(-5, 5, N)]).T
    # y_train = X_train**2 / 2 + rng.normal(0, 0.5, (N, 1))

    # データ作成(2値分類)
    X_train = np.abs(lim[1] - lim[0]) * (rng.random((N, 2)) - 0.5)
    r = 3
    mask = np.sum(X_train ** 2, axis=1) < r**2
    y_train = np.eye(2)[mask.astype(int)]

    # 学習
    batch_size = 20
    model = StandardMLP([2, 10, 10, 10, 2], task="c")
    loss_record = model.fit(X_train, y_train, batch_size=batch_size, alpha=0.01, max_iter=1000, painter=ClassificationPainter())

    # 図示
    # lossT
    plt.plot(np.log10(loss_record))
    plt.grid()
    plt.show()
    

    # affine1 = AffineLayer(1, 10)
    # relu1 = ReLuLayer()
    # bn1 = BatchNormalizationLayer(10)
    # affine2 = AffineLayer(10, 5)
    # relu2 = ReLuLayer()
    # bn2 = BatchNormalizationLayer(5)
    # affine3 = AffineLayer(5, 1)
    # # [affine1, bn1, relu1, affine2, bn2, relu2, affine3]
    # model = NeuralNetwork([affine1, relu1, affine2, relu2, affine3])
    # # model = NeuralNetwork([affine1, relu1, affine2, relu2, affine3])

    # optimizer = opt.Adam(model.params)
    # loss_record = []

    # for i in range(5000):
    #     indices = rng.choice(N, batch_size)
    #     X_batch = Variable(X[indices].T)
    #     y_batch = Variable(y[indices].T)

    #     y_hat = model.forward(X_batch)
    #     s = f.Sum()
    #     loss = s((y_batch - y_hat)**2) / 2
    #     loss_record.append(loss.data[0][0])
    #     loss.backward()
    #     optimizer.step(alpha=0.01)

    

    # X, y = optimizer._extract_batch(X, y, batch_size)
    # y_hat = nn.forward(X)
    # loss = y - y_hat
    # G = loss.backward(Graph())
    # G.show()
    # print(y_hat)








