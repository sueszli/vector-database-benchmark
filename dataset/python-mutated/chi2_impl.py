import numpy as np
from .mle import design_matrix

def lombscargle_chi2(t, y, dy, frequency, normalization='standard', fit_mean=True, center_data=True, nterms=1):
    if False:
        i = 10
        return i + 15
    "Lomb-Scargle Periodogram.\n\n    This implements a chi-squared-based periodogram, which is relatively slow\n    but useful for validating the faster algorithms in the package.\n\n    Parameters\n    ----------\n    t, y, dy : array-like\n        times, values, and errors of the data points. These should be\n        broadcastable to the same shape. None should be `~astropy.units.Quantity``.\n    frequency : array-like\n        frequencies (not angular frequencies) at which to calculate periodogram\n    normalization : str, optional\n        Normalization to use for the periodogram.\n        Options are 'standard', 'model', 'log', or 'psd'.\n    fit_mean : bool, optional\n        if True, include a constant offset as part of the model at each\n        frequency. This can lead to more accurate results, especially in the\n        case of incomplete phase coverage.\n    center_data : bool, optional\n        if True, pre-center the data by subtracting the weighted mean\n        of the input data. This is especially important if ``fit_mean = False``\n    nterms : int, optional\n        Number of Fourier terms in the fit\n\n    Returns\n    -------\n    power : array-like\n        Lomb-Scargle power associated with each frequency.\n        Units of the result depend on the normalization.\n\n    References\n    ----------\n    .. [1] M. Zechmeister and M. Kurster, A&A 496, 577-584 (2009)\n    .. [2] W. Press et al, Numerical Recipes in C (2002)\n    .. [3] Scargle, J.D. 1982, ApJ 263:835-853\n    "
    if dy is None:
        dy = 1
    (t, y, dy) = np.broadcast_arrays(t, y, dy)
    frequency = np.asarray(frequency)
    if t.ndim != 1:
        raise ValueError('t, y, dy should be one dimensional')
    if frequency.ndim != 1:
        raise ValueError('frequency should be one-dimensional')
    w = dy ** (-2.0)
    w /= w.sum()
    if center_data or fit_mean:
        yw = (y - np.dot(w, y)) / dy
    else:
        yw = y / dy
    chi2_ref = np.dot(yw, yw)

    def compute_power(f):
        if False:
            while True:
                i = 10
        X = design_matrix(t, f, dy=dy, bias=fit_mean, nterms=nterms)
        XTX = np.dot(X.T, X)
        XTy = np.dot(X.T, yw)
        return np.dot(XTy.T, np.linalg.solve(XTX, XTy))
    p = np.array([compute_power(f) for f in frequency])
    if normalization == 'psd':
        p *= 0.5
    elif normalization == 'model':
        p /= chi2_ref - p
    elif normalization == 'log':
        p = -np.log(1 - p / chi2_ref)
    elif normalization == 'standard':
        p /= chi2_ref
    else:
        raise ValueError(f"normalization='{normalization}' not recognized")
    return p