import numpy as np

def design_matrix(t, frequency, dy=None, bias=True, nterms=1):
    if False:
        while True:
            i = 10
    'Compute the Lomb-Scargle design matrix at the given frequency.\n\n    This is the matrix X such that the periodic model at the given frequency\n    can be expressed :math:`\\hat{y} = X \\theta`.\n\n    Parameters\n    ----------\n    t : array-like, shape=(n_times,)\n        times at which to compute the design matrix\n    frequency : float\n        frequency for the design matrix\n    dy : float or array-like, optional\n        data uncertainties: should be broadcastable with `t`\n    bias : bool (default=True)\n        If true, include a bias column in the matrix\n    nterms : int (default=1)\n        Number of Fourier terms to include in the model\n\n    Returns\n    -------\n    X : ndarray, shape=(n_times, n_parameters)\n        The design matrix, where n_parameters = bool(bias) + 2 * nterms\n    '
    t = np.asarray(t)
    frequency = np.asarray(frequency)
    if t.ndim != 1:
        raise ValueError('t should be one dimensional')
    if frequency.ndim != 0:
        raise ValueError('frequency must be a scalar')
    if nterms == 0 and (not bias):
        raise ValueError('cannot have nterms=0 and no bias')
    if bias:
        cols = [np.ones_like(t)]
    else:
        cols = []
    for i in range(1, nterms + 1):
        cols.append(np.sin(2 * np.pi * i * frequency * t))
        cols.append(np.cos(2 * np.pi * i * frequency * t))
    XT = np.vstack(cols)
    if dy is not None:
        XT /= dy
    return np.transpose(XT)

def periodic_fit(t, y, dy, frequency, t_fit, center_data=True, fit_mean=True, nterms=1):
    if False:
        i = 10
        return i + 15
    'Compute the Lomb-Scargle model fit at a given frequency.\n\n    Parameters\n    ----------\n    t, y, dy : float or array-like\n        The times, observations, and uncertainties to fit\n    frequency : float\n        The frequency at which to compute the model\n    t_fit : float or array-like\n        The times at which the fit should be computed\n    center_data : bool (default=True)\n        If True, center the input data before applying the fit\n    fit_mean : bool (default=True)\n        If True, include the bias as part of the model\n    nterms : int (default=1)\n        The number of Fourier terms to include in the fit\n\n    Returns\n    -------\n    y_fit : ndarray\n        The model fit evaluated at each value of t_fit\n    '
    (t, y, frequency) = map(np.asarray, (t, y, frequency))
    if dy is None:
        dy = np.ones_like(y)
    else:
        dy = np.asarray(dy)
    t_fit = np.asarray(t_fit)
    if t.ndim != 1:
        raise ValueError('t, y, dy should be one dimensional')
    if t_fit.ndim != 1:
        raise ValueError('t_fit should be one dimensional')
    if frequency.ndim != 0:
        raise ValueError('frequency should be a scalar')
    if center_data:
        w = dy ** (-2.0)
        y_mean = np.dot(y, w) / w.sum()
        y = y - y_mean
    else:
        y_mean = 0
    X = design_matrix(t, frequency, dy=dy, bias=fit_mean, nterms=nterms)
    theta_MLE = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y / dy))
    X_fit = design_matrix(t_fit, frequency, bias=fit_mean, nterms=nterms)
    return y_mean + np.dot(X_fit, theta_MLE)