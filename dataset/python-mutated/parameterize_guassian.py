from __future__ import print_function, division
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

def softplus(x):
    if False:
        print('Hello World!')
    return np.log1p(np.exp(x))
W1 = np.random.randn(4, 3)
W2 = np.random.randn(3, 2 * 2)

def forward(x, W1, W2):
    if False:
        return 10
    hidden = np.tanh(x.dot(W1))
    output = hidden.dot(W2)
    mean = output[:2]
    stddev = softplus(output[2:])
    return (mean, stddev)
x = np.random.randn(4)
(mean, stddev) = forward(x, W1, W2)
print('mean:', mean)
print('stddev:', stddev)
samples = mvn.rvs(mean=mean, cov=stddev ** 2, size=10000)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.show()