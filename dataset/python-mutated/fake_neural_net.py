from __future__ import print_function, division
from builtins import range
import numpy as np
from sklearn.svm import SVC
from util import getKaggleMNIST
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from scipy import stats
from sklearn.linear_model import LogisticRegression

class SigmoidFeaturizer:

    def __init__(self, gamma=1.0, n_components=100, method='random'):
        if False:
            while True:
                i = 10
        self.M = n_components
        self.gamma = gamma
        assert method in ('normal', 'random', 'kmeans', 'gmm')
        self.method = method

    def _subsample_data(self, X, Y, n=10000):
        if False:
            while True:
                i = 10
        if Y is not None:
            (X, Y) = shuffle(X, Y)
            return (X[:n], Y[:n])
        else:
            X = shuffle(X)
            return X[:n]

    def fit(self, X, Y=None):
        if False:
            i = 10
            return i + 15
        if self.method == 'random':
            N = len(X)
            idx = np.random.randint(N, size=self.M)
            self.samples = X[idx]
        elif self.method == 'normal':
            D = X.shape[1]
            self.samples = np.random.randn(self.M, D) / np.sqrt(D)
        elif self.method == 'kmeans':
            (X, Y) = self._subsample_data(X, Y)
            print('Fitting kmeans...')
            t0 = datetime.now()
            kmeans = KMeans(n_clusters=len(set(Y)))
            kmeans.fit(X)
            print('Finished fitting kmeans, duration:', datetime.now() - t0)
            dists = kmeans.transform(X)
            variances = dists.var(axis=1)
            idx = np.argsort(variances)
            idx = idx[:self.M]
            self.samples = X[idx]
        elif self.method == 'gmm':
            (X, Y) = self._subsample_data(X, Y)
            print('Fitting GMM')
            t0 = datetime.now()
            gmm = GaussianMixture(n_components=len(set(Y)), covariance_type='spherical', reg_covar=1e-06)
            gmm.fit(X)
            print('Finished fitting GMM, duration:', datetime.now() - t0)
            probs = gmm.predict_proba(X)
            ent = stats.entropy(probs.T)
            idx = np.argsort(-ent)
            idx = idx[:self.M]
            self.samples = X[idx]
        return self

    def transform(self, X):
        if False:
            return 10
        Z = X.dot(self.samples.T)
        return np.tanh(self.gamma * Z)

    def fit_transform(self, X, Y=None):
        if False:
            i = 10
            return i + 15
        return self.fit(X, Y).transform(X)
(Xtrain, Ytrain, Xtest, Ytest) = getKaggleMNIST()
pipeline = Pipeline([('scaler', StandardScaler()), ('sigmoid', SigmoidFeaturizer(gamma=0.05, n_components=2000, method='normal')), ('linear', LogisticRegression())])
X = np.vstack((Xtrain, Xtest))
Y = np.concatenate((Ytrain, Ytest))
scores = cross_val_score(pipeline, X, Y, cv=5)
print(scores)
print('avg:', np.mean(scores))