import numpy as np

import NeuralNetwork.function as f
import NeuralNetwork.optimizer as opt
from model import Model

Variable = f.Variable
Parameter = f.Parameter

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
        
        X_train = self._to_2d(X_train)
        y_train = self._to_2d(y_train)

        if len(X_train) != len(y_train):
            raise ValueError(f"X_train, y_trainの長さを揃えてださい. len(X_train): {len(X_train)}, len(y_train): {len(y_train)}")

        if batch_size is None:
            batch_size = min(10, len(X_train))

        optimizer = opt.Adam(self.params)

        loss_record = []

        for step in range(max_iter):

            X_batch, y_batch = self._sample(X_train, y_train, batch_size)
            X_batch, y_batch = Variable(X_batch.T), Variable(y_batch.T)

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
    pass


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