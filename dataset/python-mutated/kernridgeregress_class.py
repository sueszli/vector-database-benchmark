"""Kernel Ridge Regression for local non-parametric regression"""
import numpy as np
from scipy import spatial as ssp
import matplotlib.pylab as plt

def kernel_rbf(x, y, scale=1, **kwds):
    if False:
        while True:
            i = 10
    dist = ssp.minkowski_distance_p(x[:, np.newaxis, :], y[np.newaxis, :, :], 2)
    return np.exp(-0.5 / scale * dist)

def kernel_euclid(x, y, p=2, **kwds):
    if False:
        return 10
    return ssp.minkowski_distance(x[:, np.newaxis, :], y[np.newaxis, :, :], p)

class GaussProcess:
    """class to perform kernel ridge regression (gaussian process)

    Warning: this class is memory intensive, it creates nobs x nobs distance
    matrix and its inverse, where nobs is the number of rows (observations).
    See sparse version for larger number of observations


    Notes
    -----

    Todo:
    * normalize multidimensional x array on demand, either by var or cov
    * add confidence band
    * automatic selection or proposal of smoothing parameters

    Note: this is different from kernel smoothing regression,
       see for example https://en.wikipedia.org/wiki/Kernel_smoother

    In this version of the kernel ridge regression, the training points
    are fitted exactly.
    Needs a fast version for leave-one-out regression, for fitting each
    observation on all the other points.
    This version could be numerically improved for the calculation for many
    different values of the ridge coefficient. see also short summary by
    Isabelle Guyon (ETHZ) in a manuscript KernelRidge.pdf

    Needs verification and possibly additional statistical results or
    summary statistics for interpretation, but this is a problem with
    non-parametric, non-linear methods.

    Reference
    ---------

    Rasmussen, C.E. and C.K.I. Williams, 2006, Gaussian Processes for Machine
    Learning, the MIT Press, www.GaussianProcess.org/gpal, chapter 2

    a short summary of the kernel ridge regression is at
    http://www.ics.uci.edu/~welling/teaching/KernelsICS273B/Kernel-Ridge.pdf
    """

    def __init__(self, x, y=None, kernel=kernel_rbf, scale=0.5, ridgecoeff=1e-10, **kwds):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        x : 2d array (N,K)\n           data array of explanatory variables, columns represent variables\n           rows represent observations\n        y : 2d array (N,1) (optional)\n           endogenous variable that should be fitted or predicted\n           can alternatively be specified as parameter to fit method\n        kernel : function, default: kernel_rbf\n           kernel: (x1,x2)->kernel matrix is a function that takes as parameter\n           two column arrays and return the kernel or distance matrix\n        scale : float (optional)\n           smoothing parameter for the rbf kernel\n        ridgecoeff : float (optional)\n           coefficient that is multiplied with the identity matrix in the\n           ridge regression\n\n        Notes\n        -----\n        After initialization, kernel matrix is calculated and if y is given\n        as parameter then also the linear regression parameter and the\n        fitted or estimated y values, yest, are calculated. yest is available\n        as an attribute in this case.\n\n        Both scale and the ridge coefficient smooth the fitted curve.\n\n        '
        self.x = x
        self.kernel = kernel
        self.scale = scale
        self.ridgecoeff = ridgecoeff
        self.distxsample = kernel(x, x, scale=scale)
        self.Kinv = np.linalg.inv(self.distxsample + np.eye(*self.distxsample.shape) * ridgecoeff)
        if y is not None:
            self.y = y
            self.yest = self.fit(y)

    def fit(self, y):
        if False:
            i = 10
            return i + 15
        'fit the training explanatory variables to a sample ouput variable'
        self.parest = np.dot(self.Kinv, y)
        yhat = np.dot(self.distxsample, self.parest)
        return yhat

    def predict(self, x):
        if False:
            while True:
                i = 10
        'predict new y values for a given array of explanatory variables'
        self.xpredict = x
        distxpredict = self.kernel(x, self.x, scale=self.scale)
        self.ypredict = np.dot(distxpredict, self.parest)
        return self.ypredict

    def plot(self, y, plt=plt):
        if False:
            print('Hello World!')
        'some basic plots'
        plt.figure()
        plt.plot(self.x, self.y, 'bo-', self.x, self.yest, 'r.-')
        plt.title('sample (training) points')
        plt.figure()
        plt.plot(self.xpredict, y, 'bo-', self.xpredict, self.ypredict, 'r.-')
        plt.title('all points')

def example1():
    if False:
        i = 10
        return i + 15
    (m, k) = (500, 4)
    upper = 6
    scale = 10
    xs1a = np.linspace(1, upper, m)[:, np.newaxis]
    xs1 = xs1a * np.ones((1, 4)) + 1 / (1.0 + np.exp(np.random.randn(m, k)))
    xs1 /= np.std(xs1[::k, :], 0)
    y1true = np.sum(np.sin(xs1) + np.sqrt(xs1), 1)[:, np.newaxis]
    y1 = y1true + 0.25 * np.random.randn(m, 1)
    stride = 2
    gp1 = GaussProcess(xs1[::stride, :], y1[::stride, :], kernel=kernel_euclid, ridgecoeff=1e-10)
    yhatr1 = gp1.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1, 'bo', y1true, yhatr1, 'r.')
    plt.title('euclid kernel: true y versus noisy y and estimated y')
    plt.figure()
    plt.plot(y1, 'bo-', y1true, 'go-', yhatr1, 'r.-')
    plt.title('euclid kernel: true (green), noisy (blue) and estimated (red) ' + 'observations')
    gp2 = GaussProcess(xs1[::stride, :], y1[::stride, :], kernel=kernel_rbf, scale=scale, ridgecoeff=0.1)
    yhatr2 = gp2.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1, 'bo', y1true, yhatr2, 'r.')
    plt.title('rbf kernel: true versus noisy (blue) and estimated (red) observations')
    plt.figure()
    plt.plot(y1, 'bo-', y1true, 'go-', yhatr2, 'r.-')
    plt.title('rbf kernel: true (green), noisy (blue) and estimated (red) ' + 'observations')

def example2(m=100, scale=0.01, stride=2):
    if False:
        i = 10
        return i + 15
    upper = 6
    xs1 = np.linspace(1, upper, m)[:, np.newaxis]
    y1true = np.sum(np.sin(xs1 ** 2), 1)[:, np.newaxis] / xs1
    y1 = y1true + 0.05 * np.random.randn(m, 1)
    ridgecoeff = 1e-10
    gp1 = GaussProcess(xs1[::stride, :], y1[::stride, :], kernel=kernel_euclid, ridgecoeff=1e-10)
    yhatr1 = gp1.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1, 'bo', y1true, yhatr1, 'r.')
    plt.title('euclid kernel: true versus noisy (blue) and estimated (red) observations')
    plt.figure()
    plt.plot(y1, 'bo-', y1true, 'go-', yhatr1, 'r.-')
    plt.title('euclid kernel: true (green), noisy (blue) and estimated (red) ' + 'observations')
    gp2 = GaussProcess(xs1[::stride, :], y1[::stride, :], kernel=kernel_rbf, scale=scale, ridgecoeff=0.01)
    yhatr2 = gp2.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1, 'bo', y1true, yhatr2, 'r.')
    plt.title('rbf kernel: true versus noisy (blue) and estimated (red) observations')
    plt.figure()
    plt.plot(y1, 'bo-', y1true, 'go-', yhatr2, 'r.-')
    plt.title('rbf kernel: true (green), noisy (blue) and estimated (red) ' + 'observations')
if __name__ == '__main__':
    example2()
    example1()