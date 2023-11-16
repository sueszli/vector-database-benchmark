import numpy as np

def lombscargle_slow(t, y, dy, frequency, normalization='standard', fit_mean=True, center_data=True):
    if False:
        while True:
            i = 10
    "Lomb-Scargle Periodogram.\n\n    This is a pure-python implementation of the original Lomb-Scargle formalism\n    (e.g. [1]_, [2]_), with the addition of the floating mean (e.g. [3]_)\n\n    Parameters\n    ----------\n    t, y, dy : array-like\n        times, values, and errors of the data points. These should be\n        broadcastable to the same shape. None should be `~astropy.units.Quantity`.\n    frequency : array-like\n        frequencies (not angular frequencies) at which to calculate periodogram\n    normalization : str, optional\n        Normalization to use for the periodogram.\n        Options are 'standard', 'model', 'log', or 'psd'.\n    fit_mean : bool, optional\n        if True, include a constant offset as part of the model at each\n        frequency. This can lead to more accurate results, especially in the\n        case of incomplete phase coverage.\n    center_data : bool, optional\n        if True, pre-center the data by subtracting the weighted mean\n        of the input data. This is especially important if ``fit_mean = False``\n\n    Returns\n    -------\n    power : array-like\n        Lomb-Scargle power associated with each frequency.\n        Units of the result depend on the normalization.\n\n    References\n    ----------\n    .. [1] W. Press et al, Numerical Recipes in C (2002)\n    .. [2] Scargle, J.D. 1982, ApJ 263:835-853\n    .. [3] M. Zechmeister and M. Kurster, A&A 496, 577-584 (2009)\n    "
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
    if fit_mean or center_data:
        y = y - np.dot(w, y)
    omega = 2 * np.pi * frequency
    omega = omega.ravel()[np.newaxis, :]
    (t, y, dy, w) = (x[:, np.newaxis] for x in (t, y, dy, w))
    sin_omega_t = np.sin(omega * t)
    cos_omega_t = np.cos(omega * t)
    S2 = 2 * np.dot(w.T, sin_omega_t * cos_omega_t)
    C2 = 2 * np.dot(w.T, 0.5 - sin_omega_t ** 2)
    if fit_mean:
        S = np.dot(w.T, sin_omega_t)
        C = np.dot(w.T, cos_omega_t)
        S2 -= 2 * S * C
        C2 -= C * C - S * S
    omega_t_tau = omega * t - 0.5 * np.arctan2(S2, C2)
    sin_omega_t_tau = np.sin(omega_t_tau)
    cos_omega_t_tau = np.cos(omega_t_tau)
    Y = np.dot(w.T, y)
    wy = w * y
    YCtau = np.dot(wy.T, cos_omega_t_tau)
    YStau = np.dot(wy.T, sin_omega_t_tau)
    CCtau = np.dot(w.T, cos_omega_t_tau * cos_omega_t_tau)
    SStau = np.dot(w.T, sin_omega_t_tau * sin_omega_t_tau)
    if fit_mean:
        Ctau = np.dot(w.T, cos_omega_t_tau)
        Stau = np.dot(w.T, sin_omega_t_tau)
        YCtau -= Y * Ctau
        YStau -= Y * Stau
        CCtau -= Ctau * Ctau
        SStau -= Stau * Stau
    p = YCtau * YCtau / CCtau + YStau * YStau / SStau
    YY = np.dot(w.T, y * y)
    if normalization == 'standard':
        p /= YY
    elif normalization == 'model':
        p /= YY - p
    elif normalization == 'log':
        p = -np.log(1 - p / YY)
    elif normalization == 'psd':
        p *= 0.5 * (dy ** (-2.0)).sum()
    else:
        raise ValueError(f"normalization='{normalization}' not recognized")
    return p.ravel()