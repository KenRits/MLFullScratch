import numpy as np
from matplotlib import pyplot as plt

from NeuralNetwork.nn import StandardMLP

# 回帰
model = StandardMLP(unit_nums=[1, 16, 16, 1], task="r")

# 訓練データ作成
rng = np.random.default_rng(0)

# y = x**2 + noise というデータを100個, X \in [-5, 5]の範囲で作成
data_n = 100
X = np.linspace(-5, 5, data_n)
y_train = X**2 + rng.normal(0, 1, data_n)

# 学習
model.fit(X, y_train)

# 予測
prediction = model.predict(X)

# 訓練データをプロット
plt.scatter(X, y_train, color="blue", label="Training data")

# モデルの予測をプロット
plt.plot(X, prediction, color="red", label="MLP's prediction")

# グラフを整形
plt.legend()
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 分類
rng = np.random.default_rng(0)
model = StandardMLP(unit_nums=[2, 16, 16, 2], task="c")

# 訓練データ作成(2値分類)
data_n = 1000 # データの数
r = 3 # 原点からの距離が3未満と3以上の点でクラスを分ける
lim = (-5, 5) # データの範囲

X_train = (lim[1] - lim[0]) * (rng.random((data_n, 2)) - 0.5) # x_1, x_2 \in [-5, 5]
mask = np.sum(X_train ** 2, axis=1) < r**2
y_train = np.eye(2)[mask.astype(int)] # One-hotベクトル

# 学習
model.fit(X_train, y_train, batch_size=100, max_iter=5000)

# テストデータ作成 (データの範囲内に, 100 * 100の格子点を作成する.)
x1 = np.linspace(lim[0], lim[1], 100)
x2 = np.linspace(lim[0], lim[1], 100)
xx1, xx2 = np.meshgrid(x1, x2)
X_test = np.array([xx1.flatten(), xx2.flatten()]).T

# 予測
prediction = model.predict(X_test)

# 訓練データ図示
class1_X = X_train[mask]
class2_X = X_train[~mask]

plt.scatter(class1_X.T[0], class1_X.T[1], color="blue", label="class1")
plt.scatter(class2_X.T[0], class2_X.T[1], color="red", label="class2")

# 学習結果図示 (クラス境界の図示)
Z = np.reshape(prediction, (100, 100))
print(Z)
plt.contour(xx1, xx2, Z, colors="black", linewidths=3, levels=0)
plt.grid()
plt.legend()
plt.show()
