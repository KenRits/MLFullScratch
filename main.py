from nn import StandardMLP

import numpy as np
from matplotlib import pyplot as plt


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