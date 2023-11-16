import numpy as np
from .dfovec import dfovec

def norm(x, type=2):
    if False:
        return 10
    if type == 1:
        return np.sum(np.abs(x))
    elif type == 2:
        return np.sqrt(x ** 2)
    else:
        return max(np.abs(x))

def calfun(x, m, nprob, probtype='smooth', noise_level=0.001):
    if False:
        while True:
            i = 10
    n = len(x)
    xc = x
    if probtype == 'nondiff':
        if nprob == 8 or nprob == 9 or nprob == 13 or (nprob == 16) or (nprob == 17) or (nprob == 18):
            xc = max(x, 0)
    fvec = dfovec(m, n, xc, nprob)
    if probtype == 'noisy3':
        sigma = noise_level
        u = sigma * (-np.ones(m) + 2 * np.random.rand(m))
        fvec = fvec * (1 + u)
        y = np.sum(fvec ** 2)
    elif probtype == 'wild3':
        sigma = noise_level
        phi = 0.9 * np.sin(100 * norm(x, 1)) * np.cos(100 * norm(x, np.inf)) + 0.1 * np.cos(norm(x, 2))
        phi = phi * (4 * phi ** 2 - 3)
        y = (1 + sigma * phi) * sum(fvec ** 2)
    elif probtype == 'smooth':
        y = np.sum(fvec ** 2)
    elif probtype == 'nondiff':
        y = np.sum(np.abs(fvec))
    else:
        print(f'invalid probtype {probtype}')
        return None
    if np.isnan(y):
        return np.inf
    return y