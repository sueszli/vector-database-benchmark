from __future__ import print_function, division
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle
T = 100
x_axis = np.linspace(0, 2 * np.pi, T)
y_axis = np.sin(x_axis)
N = 30
idx = np.random.choice(T, size=N, replace=False)
Xtrain = x_axis[idx].reshape(N, 1)
Ytrain = y_axis[idx]
model = DecisionTreeRegressor()
model.fit(Xtrain, Ytrain)
prediction = model.predict(x_axis.reshape(T, 1))
print('score for 1 tree:', model.score(x_axis.reshape(T, 1), y_axis))
plt.plot(x_axis, prediction)
plt.plot(x_axis, y_axis)
plt.show()

class BaggedTreeRegressor:

    def __init__(self, B):
        if False:
            return 10
        self.B = B

    def fit(self, X, Y):
        if False:
            return 10
        N = len(X)
        self.models = []
        for b in range(self.B):
            idx = np.random.choice(N, size=N, replace=True)
            Xb = X[idx]
            Yb = Y[idx]
            model = DecisionTreeRegressor()
            model.fit(Xb, Yb)
            self.models.append(model)

    def predict(self, X):
        if False:
            print('Hello World!')
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return predictions / self.B

    def score(self, X, Y):
        if False:
            for i in range(10):
                print('nop')
        d1 = Y - self.predict(X)
        d2 = Y - Y.mean()
        return 1 - d1.dot(d1) / d2.dot(d2)
model = BaggedTreeRegressor(200)
model.fit(Xtrain, Ytrain)
print('score for bagged tree:', model.score(x_axis.reshape(T, 1), y_axis))
prediction = model.predict(x_axis.reshape(T, 1))
plt.plot(x_axis, prediction)
plt.plot(x_axis, y_axis)
plt.show()