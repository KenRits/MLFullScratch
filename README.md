# Hello, MLFullScratch
このリポジトリは, 機械学習のメジャーなモデルをフルスクラッチで(つまり機械学習系ライブラリを一切使わずに)作成するというプロジェクトのバックアップです. 
2026/8/26: 多層パーセプトロン(回帰, 多クラス分類)と, Cart(決定木)とランダムフォレストを実装しました. 

## 多層パーセプトロン
多層パーセプトロン(以降MLP)の構築方法を記します. MLPの構築のためにはnn.py内の, NeuralNetworkクラスもしくはStandardMLPクラスを使用します. 
NeuralNetworkクラスは自由度が高く汎用的なモデルを構築できますが, そのユーザインターフェースは少々煩わしいかもしれません. 一方, StandardMLPクラスはNeuralNetworkの子クラスであり, 
比較的簡単に標準的なモデルを構築することが可能です. 

### StandardMLPクラス
#### 回帰
##### インスタンスの作成
以下, 回帰タスクの場合でStandardMLPの使い方を説明します. 
```python
from nn import StandardMLP

import numpy as np
from matplotlib import pyplot as plt

model = StandardMLP(unit_nums=[1, 16, 16, 1], task="r")
```
引数unit_numsは, 各層のユニットの数を表しています. 
この例のモデルは, 1つのユニットからなる入力層, それぞれ16個のユニットを持つ2層の中間層, そして1つのユニットを持つ出力層から構成されています.
引数taskについては, "r"もしくは"c"が想定されており, それぞれ回帰タスク, 分類タスクに相当します. 
今回は回帰タスクですから, task="r"を指定します. 

##### 訓練データの作成
```python
# 訓練データ作成
rng = np.random.default_rng(0)

# y = x**2 + noise というデータを100個, X \in [-5, 5]の範囲で作成
data_n = 100
X = np.linspace(-5, 5, data_n)
y_train = X**2 + rng.normal(0, 1, data_n)
```
上の例では, $${y = x^2 + \text{noise}}$$という訓練データを作成しています. noiseは標準正規分布に従うように設定しています.

##### モデルの学習と予測
```python
# 学習
model.fit(X, y_train)

# 予測
prediction = model.predict(X)
```
model.fit(入力データ, 教師データ)によってモデルを学習します. 
学習に関するデフォルトの設定は以下の通りです. 
1. batch_size: int =None ... 一つのミニバッチに含まれる訓練データの数. Xのデータ数(=`len(X)`)が10以上のときは10, それ以外のときは`len(X)`が採用される.
2. alpha: float =0.01 ... 学習率. オプティマイザにはAdamが使用される.
3. max_iter: int =1000 ... ミニバッチの学習回数.

##### 学習結果の図示
```python
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
```
モデルの学習が成功していれば, モデルの予測(赤い線)が訓練データ(青い点)を近似するような図が表示されます. 

#### 分類
##### インスタンスの作成
```python
from nn import StandardMLP

import numpy as np
from matplotlib import pyplot as plt

model = StandardMLP(unit_nums=[2, 16, 16, 2], task="c")
```

##### 訓練データの作成
```python
# 訓練データ作成(2値分類)
rng = np.random.default_rng(0)
data_n = 1000 # データの数
r = 3         # 原点からの距離が3未満と3以上の点でクラスを分ける
lim = (-5, 5) # データの範囲

X_train = (lim[1] - lim[0]) * (rng.random((data_n, 2)) - 0.5) # x_1, x_2 \in [-5, 5]
mask = np.sum(X_train ** 2, axis=1) < r**2
y_train = np.eye(2)[mask.astype(int)] # One-hotベクトル
```
2次元の特徴量を2クラスに分類するタスクをモデルに課します. $${-5 \leq x_1 \leq 5, -5 \leq x_2 \leq 5}$$の正方形の領域から一様分布にしたがって1000点をサンプリングし, これをモデルへの入力`X_train`とします. また, 点$`(x_1, x_2)`$が原点から$`r=3`$の円の中に入っていれば, その点をクラス1, 円の外に位置していればクラス2と定義し, これを教師データ`y_train`とします. ただし`y_train`はOne-hotベクトルであり, ある訓練標本$`(x_1, x_2)`$がクラス1に所属することを`[1 0]`, クラス2に所属することを`[0 1]`で表します. 

##### モデルの学習
```python
# 学習
model.fit(X_train, y_train, batch_size=100, max_iter=5000)
```
今回は訓練標本が1000個あるので, バッチサイズもそれに合わせて100としています. また, よりよい学習のためにmax_iterを5000に設定します.
##### モデルの予測
```python
# テストデータ作成 (データの範囲内に, 100 * 100の格子点を作成する.)
x1 = np.linspace(lim[0], lim[1], 100)
x2 = np.linspace(lim[0], lim[1], 100)
xx1, xx2 = np.meshgrid(x1, x2)
X_test = np.array([xx1.flatten(), xx2.flatten()]).T

# テストデータの予測
prediction = model.predict(X_test)
```
$${-5 \leq x_1 \leq 5, -5 \leq x_2 \leq 5}$$の領域に, $${100\times100}$$個の格子点を作成し, それぞれの格子点に対して, どちらのクラスに所属するかモデルに予測させます. 
また, predictionの返り値については注意が必要です. 最初に`unit_nums=[2, 16, 16, 2]`と設定していることから, 出力層は2次元のベクトルを返しますが, `prediction`はその中で最大値をとるインデックスが格納された, 一次元のベクトルとなっています. 例えば, $3$つの訓練標本に対するモデルの出力が`np.ndarray([[-5.3, 7.2], [2, -2], [3, 1]])`であった場合, predictionは`[[1], [0], [0]]`となります. 

##### 学習結果の図示
```python
# 訓練データ図示
class1_X = X_train[mask]
class2_X = X_train[~mask]

plt.scatter(class1_X.T[0], class1_X.T[1], color="blue", label="class1")
plt.scatter(class2_X.T[0], class2_X.T[1], color="red", label="class2")

# 学習結果図示 (クラス境界の図示)
Z = np.reshape(prediction, (100, 100))
plt.contour(xx1, xx2, Z, colors="black", linewidths=3, levels=0)
plt.grid()
plt.legend()
plt.show()
```
np.countourを利用して等高線を図示します. モデルの学習がうまくいっている場合, 原点中心の半径$`3`$の円に近しい境界線が描かれます. 

## Cart(決定木), ランダムフォレスト
2026/8/26現在, 分類タスクのみに対応しています. 
モデルの使い方はほぼ変わらず, 
```python
from DecisionTree.randomForest import DecisionTreeClassifer, RandomForestClassifer

model = DecisionTreeClassifer() # Cart
model = RandomForestClassifer() # ランダムフォレスト

# 学習
model.fit(X_train, y_train)

# 予測
prediction = model.predict(X_test)
```
で使用できます. ただし, `y_train`はone-hotベクトルではなく, 単なるラベルの列ベクトルを使用します. 例として, 3つの訓練標本に対応するラベルがそれぞれ$`0, 0, 1`$であれば, 
```python
y_train = np.array([[0], [0], [1]])
```
となります. 
