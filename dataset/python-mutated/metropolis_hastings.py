import random
import numpy as np

def metropolis_hastings(func, rv, n, downsample=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Metropolis Hastings algorith\n\n    Parameters\n    ----------\n    func : callable\n        (un)normalized distribution to be sampled from\n    rv : RandomVariable\n        proposal distribution\n    n : int\n        number of samples to draw\n    downsample : int\n        downsampling factor\n\n    Returns\n    -------\n    sample : (n, ndim) ndarray\n        generated sample\n    '
    x = np.zeros((1, rv.ndim))
    sample = []
    for i in range(n * downsample):
        x_new = x + rv.draw()
        accept_proba = func(x_new) * rv.pdf(x - x_new) / (func(x) * rv.pdf(x_new - x))
        if random.random() < accept_proba:
            x = x_new
        if i % downsample == 0:
            sample.append(x[0])
    sample = np.asarray(sample)
    assert sample.shape == (n, rv.ndim), sample.shape
    return sample