import numpy as np

def forrt(X, m=None):
    if False:
        while True:
            i = 10
    '\n    RFFT with order like Munro (1976) FORTT routine.\n    '
    if m is None:
        m = len(X)
    y = np.fft.rfft(X, m) / m
    return np.r_[y.real, y[1:-1].imag]

def revrt(X, m=None):
    if False:
        print('Hello World!')
    '\n    Inverse of forrt. Equivalent to Munro (1976) REVRT routine.\n    '
    if m is None:
        m = len(X)
    i = int(m // 2 + 1)
    y = X[:i] + np.r_[0, X[i:], 0] * 1j
    return np.fft.irfft(y) * m

def silverman_transform(bw, M, RANGE):
    if False:
        for i in range(10):
            print('nop')
    '\n    FFT of Gaussian kernel following to Silverman AS 176.\n\n    Notes\n    -----\n    Underflow is intentional as a dampener.\n    '
    J = np.arange(M / 2 + 1)
    FAC1 = 2 * (np.pi * bw / RANGE) ** 2
    JFAC = J ** 2 * FAC1
    BC = 1 - 1.0 / 3 * (J * 1.0 / M * np.pi) ** 2
    FAC = np.exp(-JFAC) / BC
    kern_est = np.r_[FAC, FAC[1:-1]]
    return kern_est

def counts(x, v):
    if False:
        print('Hello World!')
    '\n    Counts the number of elements of x that fall within the grid points v\n\n    Notes\n    -----\n    Using np.digitize and np.bincount\n    '
    idx = np.digitize(x, v)
    try:
        return np.bincount(idx, minlength=len(v))
    except:
        bc = np.bincount(idx)
        return np.r_[bc, np.zeros(len(v) - len(bc))]

def kdesum(x, axis=0):
    if False:
        for i in range(10):
            print('nop')
    return np.asarray([np.sum(x[i] - x, axis) for i in range(len(x))])