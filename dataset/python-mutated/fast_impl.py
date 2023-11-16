import numpy as np
from .utils import trig_sum

def lombscargle_fast(t, y, dy, f0, df, Nf, center_data=True, fit_mean=True, normalization='standard', use_fft=True, trig_sum_kwds=None):
    if False:
        while True:
            i = 10
    'Fast Lomb-Scargle Periodogram.\n\n    This implements the Press & Rybicki method [1]_ for fast O[N log(N)]\n    Lomb-Scargle periodograms.\n\n    Parameters\n    ----------\n    t, y, dy : array-like\n        times, values, and errors of the data points. These should be\n        broadcastable to the same shape. None should be `~astropy.units.Quantity`.\n    f0, df, Nf : (float, float, int)\n        parameters describing the frequency grid, f = f0 + df * arange(Nf).\n    center_data : bool (default=True)\n        Specify whether to subtract the mean of the data before the fit\n    fit_mean : bool (default=True)\n        If True, then compute the floating-mean periodogram; i.e. let the mean\n        vary with the fit.\n    normalization : str, optional\n        Normalization to use for the periodogram.\n        Options are \'standard\', \'model\', \'log\', or \'psd\'.\n    use_fft : bool (default=True)\n        If True, then use the Press & Rybicki O[NlogN] algorithm to compute\n        the result. Otherwise, use a slower O[N^2] algorithm\n    trig_sum_kwds : dict or None, optional\n        extra keyword arguments to pass to the ``trig_sum`` utility.\n        Options are ``oversampling`` and ``Mfft``. See documentation\n        of ``trig_sum`` for details.\n\n    Returns\n    -------\n    power : ndarray\n        Lomb-Scargle power associated with each frequency.\n        Units of the result depend on the normalization.\n\n    Notes\n    -----\n    Note that the ``use_fft=True`` algorithm is an approximation to the true\n    Lomb-Scargle periodogram, and as the number of points grows this\n    approximation improves. On the other hand, for very small datasets\n    (<~50 points or so) this approximation may not be useful.\n\n    References\n    ----------\n    .. [1] Press W.H. and Rybicki, G.B, "Fast algorithm for spectral analysis\n        of unevenly sampled data". ApJ 1:338, p277, 1989\n    .. [2] M. Zechmeister and M. Kurster, A&A 496, 577-584 (2009)\n    .. [3] W. Press et al, Numerical Recipes in C (2002)\n    '
    if dy is None:
        dy = 1
    (t, y, dy) = np.broadcast_arrays(t, y, dy)
    if t.ndim != 1:
        raise ValueError('t, y, dy should be one dimensional')
    if f0 < 0:
        raise ValueError('Frequencies must be positive')
    if df <= 0:
        raise ValueError('Frequency steps must be positive')
    if Nf <= 0:
        raise ValueError('Number of frequencies must be positive')
    w = dy ** (-2.0)
    w /= w.sum()
    if center_data or fit_mean:
        y = y - np.dot(w, y)
    kwargs = dict.copy(trig_sum_kwds or {})
    kwargs.update(f0=f0, df=df, use_fft=use_fft, N=Nf)
    (Sh, Ch) = trig_sum(t, w * y, **kwargs)
    (S2, C2) = trig_sum(t, w, freq_factor=2, **kwargs)
    if fit_mean:
        (S, C) = trig_sum(t, w, **kwargs)
        tan_2omega_tau = (S2 - 2 * S * C) / (C2 - (C * C - S * S))
    else:
        tan_2omega_tau = S2 / C2
    S2w = tan_2omega_tau / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
    C2w = 1 / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
    Cw = np.sqrt(0.5) * np.sqrt(1 + C2w)
    Sw = np.sqrt(0.5) * np.sign(S2w) * np.sqrt(1 - C2w)
    YY = np.dot(w, y ** 2)
    YC = Ch * Cw + Sh * Sw
    YS = Sh * Cw - Ch * Sw
    CC = 0.5 * (1 + C2 * C2w + S2 * S2w)
    SS = 0.5 * (1 - C2 * C2w - S2 * S2w)
    if fit_mean:
        CC -= (C * Cw + S * Sw) ** 2
        SS -= (S * Cw - C * Sw) ** 2
    power = YC * YC / CC + YS * YS / SS
    if normalization == 'standard':
        power /= YY
    elif normalization == 'model':
        power /= YY - power
    elif normalization == 'log':
        power = -np.log(1 - power / YY)
    elif normalization == 'psd':
        power *= 0.5 * (dy ** (-2.0)).sum()
    else:
        raise ValueError(f"normalization='{normalization}' not recognized")
    return power