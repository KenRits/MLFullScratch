import numpy as np
from matplotlib import pyplot as plt

from NeuralNetwork.nn import StandardMLP
from DecisionTree.randomForest import DecisionTreeClassifer, RandomForestClassifer
from SVM.svm import SVC
from datasets.dataset import ToyDatasetGenerator

if __name__ == "__main__":

    # --- ニューラルネットワーク(StandardMLP)のサンプルコード. ---

    # X**2 + noise を訓練データとして生成する. 
    tdg = ToyDatasetGenerator(seed=0)
    dataset = tdg.get_square(n=100)

    # 訓練データを描画
    dataset.plot()
    plt.show()

    X_train = dataset.X
    y_train = dataset.y

    # 回帰
    model = StandardMLP(unit_nums=[1, 8, 1], task="r")

    # 学習
    loss_record = model.fit(X_train, y_train, alpha=0.2, max_iter=100, batch_size=100)

    # 損失の変遷を描画
    plt.plot(loss_record)
    plt.grid()
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()

    # 予測
    X_test = np.linspace(-3, 3, num=100)
    prediction = model.predict(X_test)

    # もう一度訓練データを描画し, axesオブジェクトを取得. その上にモデルの予測を描画していく. 
    ax = dataset.plot()

    # モデルの予測をプロット
    ax.plot(X_test, prediction, color="red", label="MLP's prediction")

    # グラフを整形
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("StandardMLP's regression")
    plt.show()

    # --- ニューラルネットワーク(分類)のサンプルコード ---
    
    dataset = tdg.get_spiral(n=100, class_n=3)

    dataset.plot()
    plt.show()

    X_train = dataset.X
    y_train = dataset.y_onehot

    model = StandardMLP([2, 8, dataset.class_n], task="c")

    # 学習
    loss_record = model.fit(X_train, y_train, batch_size=10, max_iter=1000)

    # 損失の変遷を描画
    plt.plot(np.log10(loss_record))
    plt.grid()
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()

    xx1, xx2 = dataset.meshgrid(n=100)
    X_test = np.array([xx1.ravel(), xx2.ravel()]).T

    # 予測
    prediction = model.predict(X_test)

    ax = dataset.plot()

    # 学習結果図示 (クラス境界の図示)
    zz = np.reshape(prediction, (100, 100))
    ax.contourf(xx1, xx2, zz, levels=dataset.class_n, alpha=0.5)
    ax.grid(True)
    ax.legend()
    ax.set_title("StandardMLP's decision boundary")
    plt.show()

    # Cart(決定木)のサンプルコード
    # --- 分類 --- (訓練データXは上記コードと同じものを使用)

    # y_trainはOne-hotではなく, 普通のラべルを使用
    y_train = dataset.y
    model = DecisionTreeClassifer()
    
    # 学習
    model.fit(X_train, y_train)

    # 予測
    prediction = model.predict(X_test)

    ax = dataset.plot()

    # 学習結果図示 (クラス境界の図示)
    zz = np.reshape(prediction, (100, 100))
    ax.contourf(xx1, xx2, zz, levels=dataset.class_n, alpha=0.5)
    ax.grid(True)
    ax.legend()
    ax.set_title("DecisionTreeClassifer's decision boundary")
    plt.show()


    # ランダムフォレストのサンプルコード

    model = RandomForestClassifer()
    
    # 学習
    model.fit(X_train, y_train)

    # 予測
    prediction = model.predict(X_test)

    ax = dataset.plot()

    # 学習結果図示 (クラス境界の図示)
    zz = np.reshape(prediction, (100, 100))
    ax.contourf(xx1, xx2, zz, levels=dataset.class_n, alpha=0.5)
    ax.grid(True)
    ax.legend()
    ax.set_title("RandamForest's decision boundary")
    plt.show()

    # SVMのサンプルコード

    model = SVC()
    
    # 学習
    model.fit(X_train, y_train)

    # 予測
    prediction = model.predict(X_test)

    ax = dataset.plot()

    # 学習結果図示 (クラス境界の図示)
    zz = np.reshape(prediction, (100, 100))
    ax.contourf(xx1, xx2, zz, levels=dataset.class_n, alpha=0.5)
    ax.grid(True)
    ax.legend()
    ax.set_title("SVC's decision boundary")
    plt.show()
