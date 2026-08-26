import nn
import function as f
from graph import Graph

Variable = f.Variable
Parameter = f.Parameter

import numpy as np

# Variable(1, array([[-2.12692828],
#        [-5.33969204]]))

loss_func = f.MSELoss()
X = Variable(np.array([[1, 1, 3]]))
Y = Variable(np.array([[1, 2, 3]]))
loss = loss_func(X, Y, axis=0)
print(loss)
g = loss.backward(Graph())
g.show()
print(X.grad)