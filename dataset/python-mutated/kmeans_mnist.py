from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .kmeans import plot_k_means, get_simple_data
from datetime import datetime

def get_data(limit=None):
    if False:
        while True:
            i = 10
    print('Reading in and transforming data...')
    df = pd.read_csv('../large_files/train.csv')
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    if limit is not None:
        (X, Y) = (X[:limit], Y[:limit])
    return (X, Y)

def purity2(Y, R):
    if False:
        i = 10
        return i + 15
    C = np.argmax(R, axis=1)
    N = len(Y)
    K = len(set(Y))
    total = 0.0
    for k in range(K):
        max_intersection = 0
        for j in range(K):
            intersection = ((C == k) & (Y == j)).sum()
            if intersection > max_intersection:
                max_intersection = intersection
        total += max_intersection
    return total / N

def purity(Y, R):
    if False:
        return 10
    (N, K) = R.shape
    p = 0
    for k in range(K):
        best_target = -1
        max_intersection = 0
        for j in range(K):
            intersection = R[Y == j, k].sum()
            if intersection > max_intersection:
                max_intersection = intersection
                best_target = j
        p += max_intersection
    return p / N

def DBI2(X, R):
    if False:
        for i in range(10):
            print('nop')
    (N, D) = X.shape
    (_, K) = R.shape
    sigma = np.zeros(K)
    M = np.zeros((K, D))
    assignments = np.argmax(R, axis=1)
    for k in range(K):
        Xk = X[assignments == k]
        M[k] = Xk.mean(axis=0)
        n = len(Xk)
        diffs = Xk - M[k]
        sq_diffs = diffs * diffs
        sigma[k] = np.sqrt(sq_diffs.sum() / n)
    dbi = 0
    for k in range(K):
        max_ratio = 0
        for j in range(K):
            if k != j:
                numerator = sigma[k] + sigma[j]
                denominator = np.linalg.norm(M[k] - M[j])
                ratio = numerator / denominator
                if ratio > max_ratio:
                    max_ratio = ratio
        dbi += max_ratio
    return dbi / K

def DBI(X, M, R):
    if False:
        print('Hello World!')
    (N, D) = X.shape
    (K, _) = M.shape
    sigma = np.zeros(K)
    for k in range(K):
        diffs = X - M[k]
        squared_distances = (diffs * diffs).sum(axis=1)
        weighted_squared_distances = R[:, k] * squared_distances
        sigma[k] = np.sqrt(weighted_squared_distances.sum() / R[:, k].sum())
    dbi = 0
    for k in range(K):
        max_ratio = 0
        for j in range(K):
            if k != j:
                numerator = sigma[k] + sigma[j]
                denominator = np.linalg.norm(M[k] - M[j])
                ratio = numerator / denominator
                if ratio > max_ratio:
                    max_ratio = ratio
        dbi += max_ratio
    return dbi / K

def main():
    if False:
        print('Hello World!')
    (X, Y) = get_data(10000)
    print('Number of data points:', len(Y))
    (M, R) = plot_k_means(X, len(set(Y)))
    print('Purity:', purity(Y, R))
    print('Purity 2 (hard clusters):', purity2(Y, R))
    print('DBI:', DBI(X, M, R))
    print('DBI 2 (hard clusters):', DBI2(X, R))
    for k in range(len(M)):
        im = M[k].reshape(28, 28)
        plt.imshow(im, cmap='gray')
        plt.show()
if __name__ == '__main__':
    main()