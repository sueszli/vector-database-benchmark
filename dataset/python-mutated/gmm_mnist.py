from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.mixture import GaussianMixture
from kmeans_mnist import get_data, purity, DBI

def main():
    if False:
        for i in range(10):
            print('nop')
    (X, Y) = get_data(10000)
    print('Number of data points:', len(Y))
    model = GaussianMixture(n_components=10)
    model.fit(X)
    M = model.means_
    R = model.predict_proba(X)
    print('Purity:', purity(Y, R))
    print('DBI:', DBI(X, M, R))
if __name__ == '__main__':
    main()