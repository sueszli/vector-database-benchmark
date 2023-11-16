import numpy as np
import scipy.spatial.distance as dist

class Linear(object):

    def __call__(self, x, y):
        if False:
            while True:
                i = 10
        return np.dot(x, y.T)

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'Linear kernel'

class Poly(object):

    def __init__(self, degree=2):
        if False:
            return 10
        self.degree = degree

    def __call__(self, x, y):
        if False:
            i = 10
            return i + 15
        return np.dot(x, y.T) ** self.degree

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'Poly kernel'

class RBF(object):

    def __init__(self, gamma=0.1):
        if False:
            return 10
        self.gamma = gamma

    def __call__(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        return np.exp(-self.gamma * dist.cdist(x, y) ** 2).flatten()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'RBF kernel'