"""
========================================
Plot multi-class SGD on the iris dataset
========================================

Plot decision surface of multi-class SGD on iris dataset.
The hyperplanes corresponding to the three one-versus-all (OVA) classifiers
are represented by the dashed lines.

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import SGDClassifier
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
colors = 'bry'
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std
clf = SGDClassifier(alpha=0.001, max_iter=100).fit(X, y)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(clf, X, cmap=plt.cm.Paired, ax=ax, response_method='predict', xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
plt.axis('tight')
for (i, color) in zip(clf.classes_, colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i], cmap=plt.cm.Paired, edgecolor='black', s=20)
plt.title('Decision surface of multi-class SGD')
plt.axis('tight')
(xmin, xmax) = plt.xlim()
(ymin, ymax) = plt.ylim()
coef = clf.coef_
intercept = clf.intercept_

def plot_hyperplane(c, color):
    if False:
        for i in range(10):
            print('nop')

    def line(x0):
        if False:
            i = 10
            return i + 15
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
    plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls='--', color=color)
for (i, color) in zip(clf.classes_, colors):
    plot_hyperplane(i, color)
plt.legend()
plt.show()