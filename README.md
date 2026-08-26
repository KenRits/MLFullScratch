# Hello, MLFullScratch
このリポジトリは, 機械学習のメジャーなモデルをフルスクラッチで(つまり機械学習系ライブラリを一切使わずに)作成するというプロジェクトのバックアップです. 
2026/8/26: 多層パーセプトロン(回帰, 多クラス分類)を実装しました. 

## 多層パーセプトロン
多層パーセプトロン(以降MLP)の構築方法を記します. MLPの構築のためにはnn.py内の, NeuralNetworkクラスもしくはStandardMLPクラスを使用します. 
NeuralNetworkクラスは自由度が高く汎用的なモデルを構築できますが, そのユーザインターフェースは少々煩わしいかもしれません. 一方, StandardMLPクラスはNeuralNetworkの子クラスであり, 
比較的簡単に標準的なモデルを構築することが可能です. 

### StandardMLPクラス

#### インスタンスの作成
回帰タスクの場合
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

#### 訓練データの作成
```python
# 訓練データ作成
rng = np.random.default_rng(0)

# y = x**2 + noise というデータを100個, X \in [-5, 5]の範囲で作成
data_n = 100
X = np.linspace(-5, 5, data_n)
y_train = X**2 + rng.normal(0, 1, data_n)
```
上の例では, $${y = x^2 + \text{noise}}$$という訓練データを作成しています. noiseは標準正規分布に従うように設定しています.

#### モデルの学習と予測
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
3. max_iter: =1000 ... ミニバッチの学習回数.

#### 学習結果の図示
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
