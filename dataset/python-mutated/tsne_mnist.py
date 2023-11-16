from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from util import getKaggleMNIST
import os
import sys
sys.path.append(os.path.abspath('..'))
from unsupervised_class.kmeans_mnist import purity
from sklearn.mixture import GaussianMixture

def main():
    if False:
        i = 10
        return i + 15
    (Xtrain, Ytrain, _, _) = getKaggleMNIST()
    sample_size = 1000
    X = Xtrain[:sample_size]
    Y = Ytrain[:sample_size]
    tsne = TSNE()
    Z = tsne.fit_transform(X)
    plt.scatter(Z[:, 0], Z[:, 1], s=100, c=Y, alpha=0.5)
    plt.show()
    gmm = GaussianMixture(n_components=10)
    gmm.fit(X)
    Rfull = gmm.predict_proba(X)
    print('Rfull.shape:', Rfull.shape)
    print('full purity:', purity(Y, Rfull))
    gmm.fit(Z)
    Rreduced = gmm.predict_proba(Z)
    print('reduced purity:', purity(Y, Rreduced))
if __name__ == '__main__':
    main()