import numpy as np
import heapq
import pprint
import itertools
import warnings

import NeuralNetwork.broadcast as broadcast

class Variable:
    def __init__(self, data: np.ndarray):
        # ndarrayとの四則(指数も含む)演算について, Variableの__r**__メソッドが優先されるように.
        self.__array_priority__ = 200

        data = np.atleast_2d(data)
        data.astype(np.float32)

        if data.ndim > 3:
            raise ValueError(f"3次元以上の配列はVariableとして不適切です. 2次元以下の配列, もしくは定数をdataとして入力してください. data.ndim: {data.ndim}")
        
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def add_grad(self, grad: np.ndarray):
        # 勾配は元のdataと同じ形状でなければならない. 
        assert grad.shape[0] == self.data.shape[0] and grad.shape[1] == self.data.shape[1], f"勾配は元のデータと同じ形状でなければなりません. grad.shape: {grad.shape}, self.data: {self.data}"

        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad

    def backward(self, graph=None):

        self.add_grad(np.ones_like(self.data))

        # 計算グラフ描画用
        if graph is not None:
            graph.add_node(self)

        assert self.creator is not None, f"self.creatorがNoneです. 関数によって生み出されていない{self.__class__.__name__}からbackwardを呼び出すことはできません."

        # counterは, 優先度(inp.creator.generation)が重複した場合に, 先に入れたアイテムから取り出されるようにするために入れている. 
        # これがないと, heapqはinp.creator同士を比較しようとしてエラーをはく.
        counter = itertools.count()
        funcs = [(-self.creator.generation, counter, self.creator)]
        
        while funcs:
            _, _, func = heapq.heappop(funcs)

            # 計算グラフ描画用
            if graph is not None:
                graph.add_node(func)
                graph.add_edge(func, func.out)
            
            inps = func.backward()

            for inp in inps:

                # 計算グラフ描画用
                if graph is not None:
                    graph.add_node(inp)
                    graph.add_edge(inp, func)

                if inp.creator is not None:
                    heapq.heappush(funcs, (inp.creator.generation, next(counter), inp.creator))

        return graph

    def cleargrad(self):
        self.grad = np.zeros_like(self.grad)

    def _to_variable(self, x):

        if isinstance(x, Variable):
            return x
        
        else:
            x = np.atleast_2d(x)
            if x.ndim > 2: raise ValueError(f"3次元以上の配列をVariableに変換することはできません. x.ndim: {x.ndim}")
            return Variable(x)

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    def __mul__(self, var):
        var = self._to_variable(var)
        hadamard = Hadamard()
        out = hadamard(self, var)
        return out

    def __rmul__(self, var):
        return self.__mul__(var)

    def __matmul__(self, var):
        var = self._to_variable(var)
        assert self.shape[1] == var.shape[0], f"行列の形が異なるので積を計算できません. \n self.shape: {self.shape}, var.shape: {var.shape}"
        mult = Mult()
        out = mult(self, var)
        return out

    def __rmatmul__(self, var):
        var = self._to_variable(var)
        assert self.shape[0] == var.shape[1], f"行列の形が異なるので積を計算できません. \n self.shape: {self.shape}, var.shape: {var.shape}"
        mult = Mult()
        out = mult(var, self)
        return out

    def __add__(self, var):
        var = self._to_variable(var)
        add = Add()
        out = add(self, var)
        return out

    def __radd__(self, var):
        return self.__add__(var)

    def __sub__(self, var):
        var = self._to_variable(var)
        sub = Sub()
        out = sub(self, var)
        return out

    def __rsub__(self, var):
        return self.__sub__(var)

    def __pow__(self, other):
        # otherは定数を想定. 配列や, Variableではない.
        assert not(isinstance(other, Variable) or isinstance(other, Parameter)), f"{other.__class__.__name__}は指数としてサポートされていません. 整数や浮動小数点数を使用してください."

        static_pow = StaticPower(other)
        out = static_pow(self)

        return out

    def __neg__(self):
        neg = Negative()
        out = neg(self)
        return out

    def __truediv__(self, var):
        var = self._to_variable(var)
        div = Divide()
        out = div(self, var)
        return out

    def __rtruediv__(self, var):
        var = self._to_variable(var)
        div = Divide()
        out = div(var, self)
        return out

    def __repr__(self):
        if self.data.size < 10:
            s = pprint.pformat(self.data)
        else:
            s = self.data.shape
        return f"{self.__class__.__name__}({self.generation}, {s})"
    

class Parameter(Variable):
    """最適化の対象であるVariable. OptimizerはParameterインスタンスのみを最適化の対象とする."""
    def __init__(self, data: np.ndarray):
        super().__init__(data)

class Function:
    def __init__(self):
        self._broadcast_mode = False

    def __call__(self, *inps, **kwargs):
        """入力を関数に代入し, 出力を返す.

        Args:
            *inps (tuple[Variable]): 関数への入力. 

        Returns:
            Variable: 出力.
        """

        # ブロードキャストが必要な場合, 先にブロードキャスト関数を通したVariableを取得し, それをこの関数のinpとする.
        if self._broadcast_mode:
            inps = self._broadcast(*inps)
        
        self.inps = inps
        self.generation = max([inp.generation for inp in inps])

        y = self.forward(*[i.data for i in inps], **kwargs)
        out = Variable(y)

        out.set_creator(self)

        self.out = out

        return out

    def backward(self) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__}.backwardは継承先で使用してください.")

    def forward(self) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__}.__call__は継承先で使用してください.")

    def _broadcast(self, X: Variable, Y: Variable):
    
        if X.shape != Y.shape:
        
            out_shapes = broadcast._get_broadcast_shape(X.shape, Y.shape)
            assert out_shapes is not None, f"ブロードキャストできない形状の行列が渡されました. X.shape: {X.shape}, Y.shape: {Y.shape}"
            X_out_shape, Y_out_shape = out_shapes

            if X_out_shape != X.shape:
                bt = BroadcastTo(X.shape, X_out_shape)
                X = bt(X)

            if Y_out_shape != Y.shape:
                bt = BroadcastTo(Y.shape, Y_out_shape)
                Y = bt(Y)

            return X, Y
    
        else:
            return X, Y

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.inps}\n)"


class Mult(Function):
    def __init__(self):
        super().__init__()

    def forward(self, X: np.ndarray, Y: np.ndarray):
        return X @ Y 

    def backward(self):
        """入力(Variable)のgradを更新する

        Returns:
            tuple (Variable, Variable): 入力 X, Y
        """
        X, Y = self.inps

        dloss = self.out.grad

        assert dloss is not None, "出力変数の勾配がNoneです. 逆伝播を行えません. backwardの順番を確認してください."
        
        X.add_grad(dloss @ Y.data.T)
        Y.add_grad(X.data.T @ dloss)

        return X, Y

class Add(Function):
    def __init__(self):
        super().__init__()
        self._broadcast_mode = True

    def __call__(self, *inps):
        return super().__call__(*inps)
    
    def forward(self, X: np.ndarray, Y: np.ndarray):
        return X + Y

    def backward(self):
        """入力(Variable)のgradを更新する

        Returns:
            tuple (Variable, Variable): 入力 X, Y
        """
        X, Y = self.inps

        dloss = self.out.grad

        assert dloss is not None, "出力変数の勾配がNoneです. 逆伝播を行えません. backwardの順番を確認してください."

        X.add_grad(dloss)
        Y.add_grad(dloss)

        return X, Y

class Sub(Function):
    def __init__(self):
        super().__init__()
        self._broadcast_mode = True
    
    def forward(self, X: np.ndarray, Y: np.ndarray):
        return X - Y

    def backward(self):
        """入力(Variable)のgradを更新する

        Returns:
            tuple (Variable, Variable): 入力 X, Y
        """
        X, Y = self.inps

        dloss = self.out.grad
        assert dloss is not None, "出力変数の勾配がNoneです. 逆伝播を行えません. backwardの順番を確認してください."
        
        X.add_grad(dloss)
        Y.add_grad(-dloss)

        return X, Y

class Negative(Function):
    def __init__(self):
        super().__init__()

    def forward(self, X: np.ndarray):
        return -X

    def backward(self):
        X = self.inps[0]
        dloss = self.out.grad
        assert dloss is not None, "出力変数の勾配がNoneです. 逆伝播を行えません. backwardの順番を確認してください."
        X.add_grad(-dloss)
        return (X,)

class Hadamard(Function):
    def __init__(self):
        super().__init__()
        self._broadcast_mode = True

    def forward(self, X: np.ndarray, Y: np.ndarray):
        return X * Y

    def backward(self):
        X, Y = self.inps

        dloss = self.out.grad
        assert dloss is not None, "出力変数の勾配がNoneです. 逆伝播を行えません. backwardの順番を確認してください."
        
        X.add_grad(dloss * Y.data)
        Y.add_grad(X.data * dloss)

        return X, Y

class Divide(Function):
    def __init__(self):
        super().__init__()
        self._broadcast_mode = True

    def forward(self, X: np.ndarray, Y: np.ndarray, tol=1e-8):
        # 0割を避ける.
        self._safeguard(Y, tol=tol)
        return X / Y

    def backward(self):
        X, Y = self.inps
        dloss = self.out.grad
        assert dloss is not None, "出力変数の勾配がNoneです. 逆伝播を行えません. backwardの順番を確認してください."
        
        X.add_grad(1 / Y.data)
        Y.add_grad(- X.data / self._safeguard(Y.data**2))

        return X, Y

    def _safeguard(self, Y: np.ndarray, tol=1e-8):
        mask = np.abs(Y) < tol
        Y[mask] = np.where(Y[mask] >= 0, tol, -tol)
        return Y

class Sum(Function):
    def __init__(self):
        super().__init__()

    def forward(self, X: np.ndarray, axis=None):
        assert axis in (0, 1, None), "axisは, 0, 1, Noneのいずれかを選択してください."
        return np.sum(X, axis=axis, keepdims=True)

    def backward(self):
        X = self.inps[0]
        dloss = self.out.grad
        X.add_grad(np.broadcast_to(dloss, X.shape).copy())
        return (X,)

class StaticPower(Function):
    def __init__(self, exponent: np.float32):
        super().__init__()
        self.exponent = np.float32(exponent)

    def forward(self, X: np.ndarray):
        return X ** self.exponent

    def backward(self):
        X = self.inps[0]
        dloss = self.out.grad
        assert dloss is not None, "出力変数の勾配がNoneです. 逆伝播を行えません. backwardの順番を確認してください."
        
        X.add_grad(self.exponent * X.data ** (self.exponent-1) * dloss)

        return (X,)

class BroadcastTo(Function):
    def __init__(self, inp_shape, out_shape):
        super().__init__()

        self.inp_shape, self.out_shape = inp_shape, out_shape

        # axis = 0, 1どちらの方向に引き延ばされるか記録(どちらもという場合もある)
        axes = [i for i in [0, 1] if inp_shape[i] != out_shape[i] and inp_shape[i] == 1]
        assert len(axes) > 0, f"X.shape: {inp_shape}をshape: {out_shape}にブロードキャストすることはできません."
        self.axes = axes

    def forward(self, X: np.ndarray):
        assert X.shape == self.inp_shape, f"設定されたinp_shapeと実際の入力のshapeが異なります. self.inp_shape: {self.inp_shape}, X.shape: {X.shape}"
        return np.broadcast_to(X, self.out_shape).copy()

    def backward(self):

        X = self.inps[0]
        dloss = self.out.grad
        assert dloss is not None, "出力変数の勾配がNoneです. 逆伝播を行えません. backwardの順番を確認してください."

        s = dloss
        for axis in self.axes:
            s = np.sum(s, axis=axis, keepdims=True)

        X.add_grad(s)

        return (X,)

class ReLu(Function):
    def __init__(self):
        super().__init__()

    def forward(self, X: np.ndarray):
        return np.maximum(0, X)

    def backward(self):
        X = self.inps[0]
        dloss = self.out.grad
        assert dloss is not None, "出力変数の勾配がNoneです. 逆伝播を行えません. backwardの順番を確認してください."

        X.add_grad((X.data > 0).astype(np.float64) * dloss)

        return (X,)

class SiLU(Function):
    def __init__(self):
        super().__init__()
    
    def forward(self, X: np.ndarray):
        # backwardの計算量削減のため, sigmoid(X)を記憶する.
        self._sigmoid_x = 1 / (1 + np.exp(-X))
        return X * self._sigmoid_x

    def backward(self):
        X = self.inps[0]
        dloss = self.out.grad
        assert dloss is not None, "出力変数の勾配がNoneです. 逆伝播を行えません. backwardの順番を確認してください."

        X.add_grad((self._sigmoid_x + X.data * self._sigmoid_x * (1 - self._sigmoid_x)) * dloss)

        return (X,)


class CrossEntropyLoss(Function):
    def __init__(self):
        super().__init__()
    
    def forward(self, X: np.ndarray, Y: np.ndarray, axis=0):
        if not np.all(Y >= 0):
            raise ValueError("CrossEntropyの真の分布Yはすべて0以上としてください. ")
        if not np.all(np.abs(1 - np.sum(Y, axis=axis)) <= 1e-3):
            warnings.warn(f"CrossEntropyの真の分布Yについて, axisに沿って足し合わせても1となりません. np.sum(Y, axis={axis}): {np.sum(Y, axis=axis)}")
        # オーバーフローを防止するため, Xの最大値を引いておく.
        X_max = np.max(X, axis=axis, keepdims=True)
        y_hat = (np.exp(X - X_max)) / np.sum(np.exp(X - X_max), axis=axis, keepdims=True)
        assert y_hat.shape == Y.shape, f"Yの形状と, Softmaxを通した後のy_hatの形状が等しくなるように計算を設計してください. \n y_hat.shape: {y_hat.shape}, Y.shape: {Y.shape}"

        # 逆伝播のためにsoftmaxの出力を覚えておく.
        self._y_hat = y_hat

        loss = - np.sum(Y * np.log(y_hat), keepdims=True)

        return loss

    def backward(self):
        X, Y = self.inps
        dloss = self.out.grad
        assert dloss is not None, "出力変数の勾配がNoneです. 逆伝播を行えません. backwardの順番を確認してください."

        X.add_grad((self._y_hat - Y.data) * dloss)
        Y.add_grad(np.log(self._y_hat) * dloss)

        return X, Y

    def get_softmax_out(self):
        return self._y_hat

class MSELoss(Function):
    def __init__(self):
            super().__init__()
        
    def forward(self, X: np.ndarray, Y: np.ndarray, axis=None):
        assert X.shape == Y.shape, f"Yの形状とXの形状が等しくなるように計算を設計してください. \n X.shape: {X.shape}, Y.shape: {Y.shape}"
        y_hat = X
        self._y_hat = y_hat
        loss = np.mean((y_hat - Y)**2) / 2
        return loss

    def backward(self):
        X, Y = self.inps
        dloss = self.out.grad
        assert dloss is not None, "出力変数の勾配がNoneです. 逆伝播を行えません. backwardの順番を確認してください."
    
        X.add_grad((self._y_hat - Y.data) * dloss / Y.size)
        Y.add_grad((Y.data - self._y_hat) * dloss / Y.size)
    
        return X, Y