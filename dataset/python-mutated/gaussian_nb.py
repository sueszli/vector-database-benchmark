from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from util import getKaggleMNIST

class GaussianNB(object):

    def fit(self, X, Y, smoothing=0.01):
        if False:
            return 10
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {'mean': current_x.mean(axis=0), 'var': current_x.var(axis=0) + smoothing}
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        if False:
            for i in range(10):
                print('nop')
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        if False:
            i = 10
            return i + 15
        (N, D) = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for (c, g) in iteritems(self.gaussians):
            (mean, var) = (g['mean'], g['var'])
            P[:, c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        return np.argmax(P, axis=1)
(Xtrain, Ytrain, Xtest, Ytest) = getKaggleMNIST()
model1 = GaussianNB()
model1.fit(Xtrain, Ytrain)
print('NB train score:', model1.score(Xtrain, Ytrain))
print('NB test score:', model1.score(Xtest, Ytest))
pca = PCA(n_components=50)
Ztrain = pca.fit_transform(Xtrain)
Ztest = pca.transform(Xtest)
model2 = GaussianNB()
model2.fit(Ztrain, Ytrain)
print('NB+PCA train score:', model2.score(Ztrain, Ytrain))
print('NB+PCA test score:', model2.score(Ztest, Ytest))