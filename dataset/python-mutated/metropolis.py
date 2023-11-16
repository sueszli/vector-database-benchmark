import random
import numpy as np

def metropolis(func, rv, n, downsample=1):
    if False:
        i = 10
        return i + 15
    '\n    Metropolis algorithm\n\n    Parameters\n    ----------\n    func : callable\n        (un)normalized distribution to be sampled from\n    rv : RandomVariable\n        proposal distribution which is symmetric at the origin\n    n : int\n        number of samples to draw\n    downsample : int\n        downsampling factor\n\n    Returns\n    -------\n    sample : (n, ndim) ndarray\n        generated sample\n    '
    x = np.zeros((1, rv.ndim))
    sample = []
    for i in range(n * downsample):
        x_new = x + rv.draw()
        accept_proba = func(x_new) / func(x)
        if random.random() < accept_proba:
            x = x_new
        if i % downsample == 0:
            sample.append(x[0])
    sample = np.asarray(sample)
    assert sample.shape == (n, rv.ndim), sample.shape
    return sample