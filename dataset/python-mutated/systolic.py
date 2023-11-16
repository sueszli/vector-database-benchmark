from __future__ import print_function, division
from builtins import range
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_excel('mlr02.xls', engine='xlrd')
X = df.values
plt.scatter(X[:, 1], X[:, 0])
plt.show()
plt.scatter(X[:, 2], X[:, 0])
plt.show()
df['ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'ones']]
X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r2(X, Y):
    if False:
        return 10
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Yhat = X.dot(w)
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2
print('r2 for x2 only:', get_r2(X2only, Y))
print('r2 for x3 only:', get_r2(X3only, Y))
print('r2 for both:', get_r2(X, Y))