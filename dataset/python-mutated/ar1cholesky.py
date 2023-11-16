"""
Created on Thu Oct 21 15:42:18 2010

Author: josef-pktd
"""
import numpy as np
from scipy import linalg

def tiny2zero(x, eps=1e-15):
    if False:
        while True:
            i = 10
    'replace abs values smaller than eps by zero, makes copy\n    '
    mask = np.abs(x.copy()) < eps
    x[mask] = 0
    return x
nobs = 5
autocov = 0.8 ** np.arange(nobs)
autocov = np.array([3.0, 2.0, 1.0, 0.4, 0.12, 0.016, -0.0112, 0.016, -0.0112, -0.01216, -0.007488, -0.0035584]) / 3.0
autocov = autocov[:nobs]
sigma = linalg.toeplitz(autocov)
sigmainv = linalg.inv(sigma)
c = linalg.cholesky(sigma, lower=True)
ci = linalg.cholesky(sigmainv, lower=True)
print(sigma)
print(tiny2zero(ci / ci.max()))
'this is the text book transformation'
print('coefficient for first observation', np.sqrt(1 - autocov[1] ** 2))
ci2 = ci[::-1, ::-1].T
print(tiny2zero(ci2 / ci2.max()))
print(np.dot(ci / ci.max(), np.ones(nobs)))
print(np.dot(ci2 / ci2.max(), np.ones(nobs)))