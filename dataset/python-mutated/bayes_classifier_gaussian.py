from __future__ import print_function, division
from builtins import range, input
import util
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

def clamp_sample(x):
    if False:
        i = 10
        return i + 15
    x = np.minimum(x, 1)
    x = np.maximum(x, 0)
    return x

class BayesClassifier:

    def fit(self, X, Y):
        if False:
            return 10
        self.K = len(set(Y))
        self.gaussians = []
        self.p_y = np.zeros(self.K)
        for k in range(self.K):
            Xk = X[Y == k]
            self.p_y[k] = len(Xk)
            mean = Xk.mean(axis=0)
            cov = np.cov(Xk.T)
            g = {'m': mean, 'c': cov}
            self.gaussians.append(g)
        self.p_y /= self.p_y.sum()

    def sample_given_y(self, y):
        if False:
            for i in range(10):
                print('nop')
        g = self.gaussians[y]
        return clamp_sample(mvn.rvs(mean=g['m'], cov=g['c']))

    def sample(self):
        if False:
            i = 10
            return i + 15
        y = np.random.choice(self.K, p=self.p_y)
        return clamp_sample(self.sample_given_y(y))
if __name__ == '__main__':
    (X, Y) = util.get_mnist()
    clf = BayesClassifier()
    clf.fit(X, Y)
    for k in range(clf.K):
        sample = clf.sample_given_y(k).reshape(28, 28)
        mean = clf.gaussians[k]['m'].reshape(28, 28)
        plt.subplot(1, 2, 1)
        plt.imshow(sample, cmap='gray')
        plt.title('Sample')
        plt.subplot(1, 2, 2)
        plt.imshow(mean, cmap='gray')
        plt.title('Mean')
        plt.show()
    sample = clf.sample().reshape(28, 28)
    plt.imshow(sample, cmap='gray')
    plt.title('Random Sample from Random Class')
    plt.show()