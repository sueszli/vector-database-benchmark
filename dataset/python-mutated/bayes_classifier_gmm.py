from __future__ import print_function, division
from builtins import range, input
import util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

def clamp_sample(x):
    if False:
        return 10
    x = np.minimum(x, 1)
    x = np.maximum(x, 0)
    return x

class BayesClassifier:

    def fit(self, X, Y):
        if False:
            while True:
                i = 10
        self.K = len(set(Y))
        self.gaussians = []
        self.p_y = np.zeros(self.K)
        for k in range(self.K):
            print('Fitting gmm', k)
            Xk = X[Y == k]
            self.p_y[k] = len(Xk)
            gmm = BayesianGaussianMixture(n_components=10)
            gmm.fit(Xk)
            self.gaussians.append(gmm)
        self.p_y /= self.p_y.sum()

    def sample_given_y(self, y):
        if False:
            while True:
                i = 10
        gmm = self.gaussians[y]
        sample = gmm.sample()
        mean = gmm.means_[sample[1]]
        return (clamp_sample(sample[0].reshape(28, 28)), mean.reshape(28, 28))

    def sample(self):
        if False:
            print('Hello World!')
        y = np.random.choice(self.K, p=self.p_y)
        return clamp_sample(self.sample_given_y(y))
if __name__ == '__main__':
    (X, Y) = util.get_mnist()
    clf = BayesClassifier()
    clf.fit(X, Y)
    for k in range(clf.K):
        (sample, mean) = clf.sample_given_y(k)
        plt.subplot(1, 2, 1)
        plt.imshow(sample, cmap='gray')
        plt.title('Sample')
        plt.subplot(1, 2, 2)
        plt.imshow(mean, cmap='gray')
        plt.title('Mean')
        plt.show()
    (sample, mean) = clf.sample()
    plt.subplot(1, 2, 1)
    plt.imshow(sample, cmap='gray')
    plt.title('Random Sample from Random Class')
    plt.subplot(1, 2, 2)
    plt.imshow(mean, cmap='gray')
    plt.title('Corresponding Cluster Mean')
    plt.show()