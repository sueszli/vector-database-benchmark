"""
======================
SVM with custom kernel
======================

Simple usage of Support Vector Machines to classify a sample. It will
plot the decision surface and the support vectors.

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay
iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

def my_kernel(X, Y):
    if False:
        i = 10
        return i + 15
    '\n    We create a custom kernel:\n\n                 (2  0)\n    k(X, Y) = X  (    ) Y.T\n                 (0  1)\n    '
    M = np.array([[2, 0], [0, 1.0]])
    return np.dot(np.dot(X, M), Y.T)
h = 0.02
clf = svm.SVC(kernel=my_kernel)
clf.fit(X, Y)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(clf, X, cmap=plt.cm.Paired, ax=ax, response_method='predict', plot_method='pcolormesh', shading='auto')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
plt.title('3-Class classification using Support Vector Machine with custom kernel')
plt.axis('tight')
plt.show()