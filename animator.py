from function import Variable

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np


class Painter:
    def __init__(self):
        self.images = []
        self.fig, self.ax = plt.subplots()

    def animate(self):
        assert len(self.images) > 0, "self.imagesが空配列なので, アニメーションを作成することができません. まずPainter.plot_predictionを, 継承先のクラスで実行してください."

        anime = animation.ArtistAnimation(self.fig, self.images, interval=100)
        plt.show()


class RegressionPainter(Painter):
    def __init__(self):
        super().__init__()


    def plot_prediction(self, model, X_train: np.ndarray, y_train: np.ndarray, xlim: tuple =None, ylim: tuple =None, point_n = 100):

        if X_train.shape[1] != 1 or y_train.shape[1] != 1:
            raise ValueError(f"多クラス分類の予測結果を可視化するためには, 入力, 出力どちらも1次元にしてください.\n X_train.shape[1]: {X_train.shape[1]}, y_train.shape[1]: {y_train.shape[1]}")

        MARGIN = 0.1
        if xlim is None:
            xlim = (np.min(X_train.ravel()) - MARGIN, np.max(X_train.ravel()) + MARGIN)
        
        if ylim is None:
            ylim = (np.min(y_train.ravel()) - MARGIN, np.max(y_train.ravel()) + MARGIN)
        
        if len(self.images) == 0:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.scatter(X_train.flatten(), y_train.flatten(), color="blue")
            self.ax.grid(True)

        X_test = np.array([np.linspace(xlim[0], xlim[1], point_n)])

        digits = model.forward(Variable(X_test))

        im = self.ax.plot(X_test.flatten(), digits.data.flatten(), color="red")
        self.images.append(im)




class ClassificationPainter(Painter):
    def __init__(self):
        super().__init__()


    def plot_prediction(self, model, X_train: np.ndarray, y_train: np.ndarray, xlim: tuple =None, ylim: tuple =None, point_n: int = 100):

        if X_train.shape[1] != 2:
            raise ValueError(f"多クラス分類の予測結果を可視化するためには, 特徴量の次元を2次元にしてください.\n X_train.shape[1]: {X_train.shape[1]}")

        MARGIN = 0.1
        if xlim is None:
            xlim = (np.min(X_train.T[0]) - MARGIN, np.max(X_train.T[0]) + MARGIN)

        if ylim is None:
            ylim = (np.min(X_train.T[1]) - MARGIN, np.max(X_train.T[1]) + MARGIN)
        
        if len(self.images) == 0:

            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.grid(True)

            # 総クラス数
            class_n = y_train.shape[1]
            labels = np.argmax(y_train, axis=1)
    
            # 訓練標本の散布図を作成.
            for c in range(class_n):
                X_in_the_class = X_train[labels==c]
                self.ax.scatter(X_in_the_class.T[0], X_in_the_class.T[1])

        x_axis = np.linspace(xlim[0], xlim[1], point_n)
        y_axis = np.linspace(ylim[0], ylim[1], point_n)

        xx, yy = np.meshgrid(x_axis, y_axis)

        X_test = np.array([xx.flatten(), yy.flatten()])

        digits = model.forward(Variable(X_test))

        zz = np.argmax(digits.data, axis=0).reshape((point_n, point_n))
        
        cs = self.ax.contour(xx, yy, zz)

        self.images.append([cs])



        

        





        