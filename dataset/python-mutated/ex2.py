from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt

def sampleY(n=1000):
    if False:
        return 10
    X = np.random.random(n)
    Y = X.sum()
    return Y
N = 1000
Y_samples = np.zeros(N)
for i in range(N):
    Y_samples[i] = sampleY()
plt.hist(Y_samples, bins=20)
plt.show()