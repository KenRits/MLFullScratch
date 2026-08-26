# nn.py
## StandardMLP
```python
N = 1000

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
# loss
plt.plot(np.log10(loss_record))
plt.grid()
plt.show()
```