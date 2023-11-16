import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mla.kmeans import KMeans
from mla.gaussian_mixture import GaussianMixture
random.seed(1)
np.random.seed(6)

def make_clusters(skew=True, *arg, **kwargs):
    if False:
        i = 10
        return i + 15
    (X, y) = datasets.make_blobs(*arg, **kwargs)
    if skew:
        nrow = X.shape[1]
        for i in np.unique(y):
            X[y == i] = X[y == i].dot(np.random.random((nrow, nrow)) - 0.5)
    return (X, y)

def KMeans_and_GMM(K):
    if False:
        i = 10
        return i + 15
    COLOR = 'bgrcmyk'
    (X, y) = make_clusters(skew=True, n_samples=1500, centers=K)
    (_, axes) = plt.subplots(1, 3)
    axes[0].scatter(X[:, 0], X[:, 1], c=[COLOR[int(assignment)] for assignment in y])
    axes[0].set_title('Ground Truth')
    kmeans = KMeans(K=K, init='++')
    kmeans.fit(X)
    kmeans.predict()
    axes[1].set_title('KMeans')
    kmeans.plot(ax=axes[1], holdon=True)
    gmm = GaussianMixture(K=K, init='kmeans')
    gmm.fit(X)
    axes[2].set_title('Gaussian Mixture')
    gmm.plot(ax=axes[2])
if __name__ == '__main__':
    KMeans_and_GMM(4)