from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np


class Animator:
    def __init__(self, iter_interval=100):
        self.images = []
        self.fig, self.ax = plt.subplots()
        self.counter = 0
        self.MARGIN = 0.1
        self.iter_interval = iter_interval

    def animate(self):
        assert len(self.images) > 0, "self.imagesが空配列なので, アニメーションを作成することができません. まずPainter.plot_predictionを, 継承先のクラスで実行してください."

        anime = animation.ArtistAnimation(self.fig, self.images, interval=100)
        plt.show()

    def _plot_train_data(self):
        raise NotImplemented(f"{self.__class__.__name___}.plot_train_dataは継承先で使用してください.")

    def _check_train_data(self):
        raise NotImplemented(f"{self.__class__.__name___}.check_train_dataは継承先で使用してください.")

    def _init_image(self, X_train: np.ndarray, y_train: np.ndarray, xlim: tuple, ylim: tuple):
        """アニメーションにおいて変わらない要素をプロットしておく.

        Args:
            X_train (np.ndarray): 1次元訓練データ shape: (データ数, 1)
            y_train (np.ndarray): 1次元教師データ shape: (データ数, 1)
            xlim (tuple): 図の横軸の描画範囲
            ylim (tuple): 図の縦軸の描画範囲
        """
        self._check_train_data(X_train, y_train)
        self._set_image_range(X_train, y_train, xlim, ylim)
        self._plot_train_data(X_train, y_train)
        self.ax.grid()

    def get_iter_interval(self):
        return self.iter_interval

class RegressionAnimator(Animator):
    def __init__(self, iter_interval=100):
        super().__init__(iter_interval=iter_interval)

    def _check_train_data(self, X_train: np.ndarray, y_train: np.ndarray):
        
        # 1次元 -> 1次元の回帰タスクでないと可視化できない. 
        if X_train.shape[1] != 1 or y_train.shape[1] != 1:
            raise ValueError(f"回帰タスクの予測結果を可視化するためには, 入力, 出力どちらも1次元にしてください.\n X_train.shape[1]: {X_train.shape[1]}, y_train.shape[1]: {y_train.shape[1]}")

    def _set_image_range(self, X_train: np.ndarray, y_train: np.ndarray, xlim: tuple, ylim: tuple):
        # 画像として表示する範囲を決定.
        if xlim is None:
            xlim = (np.min(X_train.ravel()) - self.MARGIN, np.max(X_train.ravel()) + self.MARGIN)
        
        if ylim is None:
            ylim = (np.min(y_train.ravel()) - self.MARGIN, np.max(y_train.ravel()) + self.MARGIN)

        self.xlim = xlim
        self.ylim = ylim
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def _plot_train_data(self, X_train: np.ndarray, y_train: np.ndarray):
        self.ax.scatter(X_train.flatten(), y_train.flatten(), color="blue")

    def _init_image(self, X_train, y_train, xlim, ylim):
        super()._init_image(X_train, y_train, xlim, ylim)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

    def plot_prediction(self, model, X_train: np.ndarray, y_train: np.ndarray, xlim: tuple =None, ylim: tuple =None, point_n = 100):

        # 最初のプロットでは, 訓練データを散布図として描画しておく. 
        if len(self.images) == 0:
            self._init_image(X_train, y_train, xlim, ylim)

        X_test = np.array([np.linspace(self.xlim[0], self.xlim[1], point_n)]).T

        prediction = model.predict(X_test)

        im = self.ax.plot(X_test.flatten(), prediction.flatten(), color="red")

        self.images.append(im)

class ClassificationAnimator(Animator):
    def __init__(self, iter_interval=100):
        super().__init__(iter_interval=iter_interval)

    def _check_train_data(self, X_train: np.ndarray, _):
        if X_train.shape[1] != 2:
            raise ValueError(f"多クラス分類の予測結果を可視化するためには, 特徴量の次元を2次元にしてください.\n X_train.shape[1]: {X_train.shape[1]}")

    def _set_image_range(self, X_train: np.ndarray, _, xlim: tuple, ylim: tuple):

        # 画像として表示する範囲を決定.
        if xlim is None:
            xlim = (np.min(X_train.T[0]) - self.MARGIN, np.max(X_train.T[0]) + self.MARGIN)
        
        if ylim is None:
            ylim = (np.min(X_train.T[1]) - self.MARGIN, np.max(X_train.T[1]) + self.MARGIN)

        self.xlim = xlim
        self.ylim = ylim
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def _plot_train_data(self, X_train: np.ndarray, y_train: np.ndarray):
        # 総クラス数
        class_n = y_train.shape[1]
        # 等高線用にone-hotベクトルを, そのインデックスに変換. Ex. [[0, 1], [1, 0], [0, 1]] -> [1 0 1]
        labels = np.argmax(y_train, axis=1)

        # 訓練標本の散布図を作成.
        for c in range(class_n):
            X_in_the_class = X_train[labels==c]
            self.ax.scatter(X_in_the_class.T[0], X_in_the_class.T[1])

    def _init_image(self, X_train, y_train, xlim, ylim):
        super()._init_image(X_train, y_train, xlim, ylim)
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        
    def plot_prediction(self, model, X_train: np.ndarray, y_train: np.ndarray, xlim: tuple =None, ylim: tuple =None, point_n: int = 100):

        if len(self.images) == 0:
            self._init_image(X_train, y_train, xlim, ylim)

        x_axis = np.linspace(self.xlim[0], self.xlim[1], point_n)
        y_axis = np.linspace(self.ylim[0], self.ylim[1], point_n)
        xx, yy = np.meshgrid(x_axis, y_axis)

        X_test = np.array([xx.flatten(), yy.flatten()]).T

        prediction = model.predict(X_test)

        zz = prediction.reshape((point_n, point_n))
        
        cs = self.ax.contour(xx, yy, zz)

        self.images.append([cs])