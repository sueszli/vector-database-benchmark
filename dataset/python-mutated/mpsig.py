"""
Some signal functions implemented using mpmath.
"""
try:
    import mpmath
except ImportError:
    mpmath = None

def _prod(seq):
    if False:
        i = 10
        return i + 15
    'Returns the product of the elements in the sequence `seq`.'
    p = 1
    for elem in seq:
        p *= elem
    return p

def _relative_degree(z, p):
    if False:
        return 10
    '\n    Return relative degree of transfer function from zeros and poles.\n\n    This is simply len(p) - len(z), which must be nonnegative.\n    A ValueError is raised if len(p) < len(z).\n    '
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError('Improper transfer function. Must have at least as many poles as zeros.')
    return degree

def _zpkbilinear(z, p, k, fs):
    if False:
        i = 10
        return i + 15
    'Bilinear transformation to convert a filter from analog to digital.'
    degree = _relative_degree(z, p)
    fs2 = 2 * fs
    z_z = [(fs2 + z1) / (fs2 - z1) for z1 in z]
    p_z = [(fs2 + p1) / (fs2 - p1) for p1 in p]
    z_z.extend([-1] * degree)
    numer = _prod((fs2 - z1 for z1 in z))
    denom = _prod((fs2 - p1 for p1 in p))
    k_z = k * numer / denom
    return (z_z, p_z, k_z.real)

def _zpklp2lp(z, p, k, wo=1):
    if False:
        while True:
            i = 10
    'Transform a lowpass filter to a different cutoff frequency.'
    degree = _relative_degree(z, p)
    z_lp = [wo * z1 for z1 in z]
    p_lp = [wo * p1 for p1 in p]
    k_lp = k * wo ** degree
    return (z_lp, p_lp, k_lp)

def _butter_analog_poles(n):
    if False:
        i = 10
        return i + 15
    "\n    Poles of an analog Butterworth lowpass filter.\n\n    This is the same calculation as scipy.signal.buttap(n) or\n    scipy.signal.butter(n, 1, analog=True, output='zpk'), but mpmath is used,\n    and only the poles are returned.\n    "
    poles = [-mpmath.exp(1j * mpmath.pi * k / (2 * n)) for k in range(-n + 1, n, 2)]
    return poles

def butter_lp(n, Wn):
    if False:
        i = 10
        return i + 15
    "\n    Lowpass Butterworth digital filter design.\n\n    This computes the same result as scipy.signal.butter(n, Wn, output='zpk'),\n    but it uses mpmath, and the results are returned in lists instead of NumPy\n    arrays.\n    "
    zeros = []
    poles = _butter_analog_poles(n)
    k = 1
    fs = 2
    warped = 2 * fs * mpmath.tan(mpmath.pi * Wn / fs)
    (z, p, k) = _zpklp2lp(zeros, poles, k, wo=warped)
    (z, p, k) = _zpkbilinear(z, p, k, fs=fs)
    return (z, p, k)

def zpkfreqz(z, p, k, worN=None):
    if False:
        return 10
    '\n    Frequency response of a filter in zpk format, using mpmath.\n\n    This is the same calculation as scipy.signal.freqz, but the input is in\n    zpk format, the calculation is performed using mpath, and the results are\n    returned in lists instead of NumPy arrays.\n    '
    if worN is None or isinstance(worN, int):
        N = worN or 512
        ws = [mpmath.pi * mpmath.mpf(j) / N for j in range(N)]
    else:
        ws = worN
    h = []
    for wk in ws:
        zm1 = mpmath.exp(1j * wk)
        numer = _prod([zm1 - t for t in z])
        denom = _prod([zm1 - t for t in p])
        hk = k * numer / denom
        h.append(hk)
    return (ws, h)