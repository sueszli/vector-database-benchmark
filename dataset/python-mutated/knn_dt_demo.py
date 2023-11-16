from __future__ import print_function, division
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.utils import shuffle
N = 20
Ntrain = 12
X = np.linspace(0, 2 * np.pi, N).reshape(N, 1)
Y = np.sin(3 * X)
(X, Y) = shuffle(X, Y)
Xtrain = X[:Ntrain]
Ytrain = Y[:Ntrain]
model = DecisionTreeRegressor()
model.fit(Xtrain, Ytrain)
T = 50
Xaxis = np.linspace(0, 2 * np.pi, T)
Yaxis = np.sin(3 * Xaxis)
plt.scatter(Xtrain, Ytrain, s=50, alpha=0.7, c='blue')
plt.scatter(Xtrain, model.predict(Xtrain.reshape(Ntrain, 1)), s=50, alpha=0.7, c='green')
plt.title('decision tree - low bias, high variance')
plt.plot(Xaxis, Yaxis)
plt.plot(Xaxis, model.predict(Xaxis.reshape(T, 1)))
plt.show()
model = DecisionTreeRegressor(max_depth=1)
model.fit(Xtrain, Ytrain)
plt.scatter(Xtrain, Ytrain, s=50, alpha=0.7, c='blue')
plt.scatter(Xtrain, model.predict(Xtrain.reshape(Ntrain, 1)), s=50, alpha=0.7, c='green')
plt.plot(Xaxis, Yaxis)
plt.plot(Xaxis, model.predict(Xaxis.reshape(T, 1)))
plt.title('decision tree - high bias, low variance')
plt.show()
model = KNeighborsRegressor(n_neighbors=1)
model.fit(Xtrain, Ytrain)
plt.scatter(Xtrain, Ytrain, s=50, alpha=0.7, c='blue')
plt.scatter(Xtrain, model.predict(Xtrain.reshape(Ntrain, 1)), s=50, alpha=0.7, c='green')
plt.plot(Xaxis, Yaxis)
plt.plot(Xaxis, model.predict(Xaxis.reshape(T, 1)))
plt.title('knn - low bias, high variance')
plt.show()
model = KNeighborsRegressor(n_neighbors=10)
model.fit(Xtrain, Ytrain)
plt.scatter(Xtrain, Ytrain, s=50, alpha=0.7, c='blue')
plt.scatter(Xtrain, model.predict(Xtrain.reshape(Ntrain, 1)), s=50, alpha=0.7, c='green')
plt.plot(Xaxis, Yaxis)
plt.plot(Xaxis, model.predict(Xaxis.reshape(T, 1)))
plt.title('knn - high bias, low variance')
plt.show()
N = 100
D = 2
X = np.random.randn(N, D)
X[:N // 2] += np.array([1, 1])
X[N // 2:] += np.array([-1, -1])
Y = np.array([0] * (N // 2) + [1] * (N // 2))

def plot_decision_boundary(X, model):
    if False:
        i = 10
        return i + 15
    h = 0.02
    (x_min, x_max) = (X[:, 0].min() - 1, X[:, 0].max() + 1)
    (y_min, y_max) = (X[:, 1].min() - 1, X[:, 1].max() + 1)
    (xx, yy) = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], s=50, c=Y, alpha=0.7)
plt.show()
model = DecisionTreeClassifier()
model.fit(X, Y)
plt.scatter(X[:, 0], X[:, 1], s=50, c=Y, alpha=0.7)
plot_decision_boundary(X, model)
plt.title('dt - low bias, high variance')
plt.show()
model = DecisionTreeClassifier(max_depth=2)
model.fit(X, Y)
plt.scatter(X[:, 0], X[:, 1], s=50, c=Y, alpha=0.7)
plot_decision_boundary(X, model)
plt.title('dt - high bias, low variance')
plt.show()
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, Y)
plt.scatter(X[:, 0], X[:, 1], s=50, c=Y, alpha=0.7)
plot_decision_boundary(X, model)
plt.title('knn - low bias, high variance')
plt.show()
model = KNeighborsClassifier(n_neighbors=20)
model.fit(X, Y)
plt.scatter(X[:, 0], X[:, 1], s=50, c=Y, alpha=0.7)
plot_decision_boundary(X, model)
plt.title('knn - high bias, low variance')
plt.show()