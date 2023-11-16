"""
Statistic functions used in `~astropy.modeling.fitting`.
"""
import numpy as np
__all__ = ['leastsquare', 'leastsquare_1d', 'leastsquare_2d', 'leastsquare_3d']

def leastsquare(measured_vals, updated_model, weights, *x):
    if False:
        print('Hello World!')
    'Least square statistic, with optional weights, in N-dimensions.\n\n    Parameters\n    ----------\n    measured_vals : ndarray or sequence\n        Measured data values. Will be cast to array whose\n        shape must match the array-cast of the evaluated model.\n    updated_model : :class:`~astropy.modeling.Model` instance\n        Model with parameters set by the current iteration of the optimizer.\n        when evaluated on "x", must return array of shape "measured_vals"\n    weights : ndarray or None\n        Array of weights to apply to each residual.\n    *x : ndarray\n        Independent variables on which to evaluate the model.\n\n    Returns\n    -------\n    res : float\n        The sum of least squares.\n\n    See Also\n    --------\n    :func:`~astropy.modeling.statistic.leastsquare_1d`\n    :func:`~astropy.modeling.statistic.leastsquare_2d`\n    :func:`~astropy.modeling.statistic.leastsquare_3d`\n\n    Notes\n    -----\n    Models in :mod:`~astropy.modeling` have broadcasting rules that try to\n    match inputs with outputs with Model shapes. Numpy arrays have flexible\n    broadcasting rules, so mismatched shapes can often be made compatible. To\n    ensure data matches the model we must perform shape comparison and leverage\n    the Numpy arithmetic functions. This can obfuscate arithmetic computation\n    overrides, like with Quantities. Implement a custom statistic for more\n    direct control.\n\n    '
    model_vals = updated_model(*x)
    if np.shape(model_vals) != np.shape(measured_vals):
        raise ValueError(f'Shape mismatch between model ({np.shape(model_vals)}) and measured ({np.shape(measured_vals)})')
    if weights is None:
        weights = 1.0
    return np.sum(np.square(weights * np.subtract(model_vals, measured_vals)))

def leastsquare_1d(measured_vals, updated_model, weights, x):
    if False:
        print('Hello World!')
    '\n    Least square statistic with optional weights.\n    Safer than the general :func:`~astropy.modeling.statistic.leastsquare`\n    for 1D models by avoiding numpy methods that support broadcasting.\n\n    Parameters\n    ----------\n    measured_vals : ndarray\n        Measured data values.\n    updated_model : `~astropy.modeling.Model`\n        Model with parameters set by the current iteration of the optimizer.\n    weights : ndarray or None\n        Array of weights to apply to each residual.\n    x : ndarray\n        Independent variable "x" on which to evaluate the model.\n\n    Returns\n    -------\n    res : float\n        The sum of least squares.\n\n    See Also\n    --------\n    :func:`~astropy.modeling.statistic.leastsquare`\n\n    '
    model_vals = updated_model(x)
    if weights is None:
        return np.sum((model_vals - measured_vals) ** 2)
    return np.sum((weights * (model_vals - measured_vals)) ** 2)

def leastsquare_2d(measured_vals, updated_model, weights, x, y):
    if False:
        print('Hello World!')
    '\n    Least square statistic with optional weights.\n    Safer than the general :func:`~astropy.modeling.statistic.leastsquare`\n    for 2D models by avoiding numpy methods that support broadcasting.\n\n    Parameters\n    ----------\n    measured_vals : ndarray\n        Measured data values.\n    updated_model : `~astropy.modeling.Model`\n        Model with parameters set by the current iteration of the optimizer.\n    weights : ndarray or None\n        Array of weights to apply to each residual.\n    x : ndarray\n        Independent variable "x" on which to evaluate the model.\n    y : ndarray\n        Independent variable "y" on which to evaluate the model.\n\n    Returns\n    -------\n    res : float\n        The sum of least squares.\n\n    See Also\n    --------\n    :func:`~astropy.modeling.statistic.leastsquare`\n\n    '
    model_vals = updated_model(x, y)
    if weights is None:
        return np.sum((model_vals - measured_vals) ** 2)
    return np.sum((weights * (model_vals - measured_vals)) ** 2)

def leastsquare_3d(measured_vals, updated_model, weights, x, y, z):
    if False:
        i = 10
        return i + 15
    '\n    Least square statistic with optional weights.\n    Safer than the general :func:`~astropy.modeling.statistic.leastsquare`\n    for 3D models by avoiding numpy methods that support broadcasting.\n\n    Parameters\n    ----------\n    measured_vals : ndarray\n        Measured data values.\n    updated_model : `~astropy.modeling.Model`\n        Model with parameters set by the current iteration of the optimizer.\n    weights : ndarray or None\n        Array of weights to apply to each residual.\n    x : ndarray\n        Independent variable "x" on which to evaluate the model.\n    y : ndarray\n        Independent variable "y" on which to evaluate the model.\n    z : ndarray\n        Independent variable "z" on which to evaluate the model.\n\n    Returns\n    -------\n    res : float\n        The sum of least squares.\n\n    See Also\n    --------\n    :func:`~astropy.modeling.statistic.leastsquare`\n\n    '
    model_vals = updated_model(x, y, z)
    if weights is None:
        return np.sum((model_vals - measured_vals) ** 2)
    return np.sum((weights * (model_vals - measured_vals)) ** 2)