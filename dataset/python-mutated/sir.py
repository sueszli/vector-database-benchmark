import numpy as np

def sir(func, rv, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    sampling-importance-resampling\n\n    Parameters\n    ----------\n    func : callable\n        (un)normalized distribution to be sampled from\n    rv : RandomVariable\n        distribution to generate sample\n    n : int\n        number of samples to draw\n\n    Returns\n    -------\n    sample : (n, ndim) ndarray\n        generated sample\n    '
    assert hasattr(rv, 'draw'), 'the distribution has no method to draw random samples'
    sample_candidate = rv.draw(n * 10)
    weight = np.squeeze(func(sample_candidate) / rv.pdf(sample_candidate))
    assert weight.shape == (n * 10,), weight.shape
    weight /= np.sum(weight)
    index = np.random.choice(n * 10, n, p=weight)
    sample = sample_candidate[index]
    return sample