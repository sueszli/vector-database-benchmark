import numpy as np

def lombscargle_scipy(t, y, frequency, normalization='standard', center_data=True):
    if False:
        for i in range(10):
            print('nop')
    "Lomb-Scargle Periodogram.\n\n    This is a wrapper of ``scipy.signal.lombscargle`` for computation of the\n    Lomb-Scargle periodogram. This is a relatively fast version of the naive\n    O[N^2] algorithm, but cannot handle heteroskedastic errors.\n\n    Parameters\n    ----------\n    t, y : array-like\n        times, values, and errors of the data points. These should be\n        broadcastable to the same shape. None should be `~astropy.units.Quantity`.\n    frequency : array-like\n        frequencies (not angular frequencies) at which to calculate periodogram\n    normalization : str, optional\n        Normalization to use for the periodogram.\n        Options are 'standard', 'model', 'log', or 'psd'.\n    center_data : bool, optional\n        if True, pre-center the data by subtracting the weighted mean\n        of the input data.\n\n    Returns\n    -------\n    power : array-like\n        Lomb-Scargle power associated with each frequency.\n        Units of the result depend on the normalization.\n\n    References\n    ----------\n    .. [1] M. Zechmeister and M. Kurster, A&A 496, 577-584 (2009)\n    .. [2] W. Press et al, Numerical Recipes in C (2002)\n    .. [3] Scargle, J.D. 1982, ApJ 263:835-853\n    "
    try:
        from scipy import signal
    except ImportError:
        raise ImportError('scipy must be installed to use lombscargle_scipy')
    (t, y) = np.broadcast_arrays(t, y)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    frequency = np.asarray(frequency, dtype=float)
    if t.ndim != 1:
        raise ValueError('t, y, dy should be one dimensional')
    if frequency.ndim != 1:
        raise ValueError('frequency should be one-dimensional')
    if center_data:
        y = y - y.mean()
    p = signal.lombscargle(t, y, 2 * np.pi * frequency)
    if normalization == 'psd':
        pass
    elif normalization == 'standard':
        p *= 2 / (t.size * np.mean(y ** 2))
    elif normalization == 'log':
        p = -np.log(1 - 2 * p / (t.size * np.mean(y ** 2)))
    elif normalization == 'model':
        p /= 0.5 * t.size * np.mean(y ** 2) - p
    else:
        raise ValueError(f"normalization='{normalization}' not recognized")
    return p