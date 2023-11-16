from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
from kmeans import plot_k_means

def donut():
    if False:
        for i in range(10):
            print('nop')
    N = 1000
    D = 2
    R_inner = 5
    R_outer = 10
    R1 = np.random.randn(N // 2) + R_inner
    theta = 2 * np.pi * np.random.random(N // 2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T
    R2 = np.random.randn(N // 2) + R_outer
    theta = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T
    X = np.concatenate([X_inner, X_outer])
    return X

def main():
    if False:
        i = 10
        return i + 15
    X = donut()
    plot_k_means(X, 2, beta=0.1, show_plots=True)
    X = np.zeros((1000, 2))
    X[:500, :] = np.random.multivariate_normal([0, 0], [[1, 0], [0, 20]], 500)
    X[500:, :] = np.random.multivariate_normal([5, 0], [[1, 0], [0, 20]], 500)
    plot_k_means(X, 2, beta=0.1, show_plots=True)
    X = np.zeros((1000, 2))
    X[:950, :] = np.array([0, 0]) + np.random.randn(950, 2)
    X[950:, :] = np.array([3, 0]) + np.random.randn(50, 2)
    plot_k_means(X, 2, show_plots=True)
if __name__ == '__main__':
    main()