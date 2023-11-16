"""

Special functions for copulas not available in scipy

Created on Jan. 27, 2023
"""
import numpy as np
from scipy.special import factorial

class Sterling1:
    """Stirling numbers of the first kind
    """

    def __init__(self):
        if False:
            return 10
        self._cache = {}

    def __call__(self, n, k):
        if False:
            i = 10
            return i + 15
        key = str(n) + ',' + str(k)
        if key in self._cache.keys():
            return self._cache[key]
        if n == k == 0:
            return 1
        if n > 0 and k == 0:
            return 0
        if k > n:
            return 0
        result = sterling1(n - 1, k - 1) + (n - 1) * sterling1(n - 1, k)
        self._cache[key] = result
        return result

    def clear_cache(self):
        if False:
            i = 10
            return i + 15
        'clear cache of Sterling numbers\n        '
        self._cache = {}
sterling1 = Sterling1()

class Sterling2:
    """Stirling numbers of the second kind
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._cache = {}

    def __call__(self, n, k):
        if False:
            while True:
                i = 10
        key = str(n) + ',' + str(k)
        if key in self._cache.keys():
            return self._cache[key]
        if n == k == 0:
            return 1
        if n > 0 and k == 0 or (n == 0 and k > 0):
            return 0
        if n == k:
            return 1
        if k > n:
            return 0
        result = k * sterling2(n - 1, k) + sterling2(n - 1, k - 1)
        self._cache[key] = result
        return result

    def clear_cache(self):
        if False:
            return 10
        'clear cache of Sterling numbers\n        '
        self._cache = {}
sterling2 = Sterling2()

def li3(z):
    if False:
        print('Hello World!')
    'Polylogarithm for negative integer order -3\n\n    Li(-3, z)\n    '
    return z * (1 + 4 * z + z ** 2) / (1 - z) ** 4

def li4(z):
    if False:
        for i in range(10):
            print('nop')
    'Polylogarithm for negative integer order -4\n\n    Li(-4, z)\n    '
    return z * (1 + z) * (1 + 10 * z + z ** 2) / (1 - z) ** 5

def lin(n, z):
    if False:
        return 10
    'Polylogarithm for negative integer order -n\n\n    Li(-n, z)\n\n    https://en.wikipedia.org/wiki/Polylogarithm#Particular_values\n    '
    if np.size(z) > 1:
        z = np.array(z)[..., None]
    k = np.arange(n + 1)
    st2 = np.array([sterling2(n + 1, ki + 1) for ki in k])
    res = (-1) ** (n + 1) * np.sum(factorial(k) * st2 * (-1 / (1 - z)) ** (k + 1), axis=-1)
    return res