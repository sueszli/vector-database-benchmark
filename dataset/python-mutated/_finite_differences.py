from numpy import arange, newaxis, hstack, prod, array

def _central_diff_weights(Np, ndiv=1):
    if False:
        while True:
            i = 10
    "\n    Return weights for an Np-point central derivative.\n\n    Assumes equally-spaced function points.\n\n    If weights are in the vector w, then\n    derivative is w[0] * f(x-ho*dx) + ... + w[-1] * f(x+h0*dx)\n\n    Parameters\n    ----------\n    Np : int\n        Number of points for the central derivative.\n    ndiv : int, optional\n        Number of divisions. Default is 1.\n\n    Returns\n    -------\n    w : ndarray\n        Weights for an Np-point central derivative. Its size is `Np`.\n\n    Notes\n    -----\n    Can be inaccurate for a large number of points.\n\n    Examples\n    --------\n    We can calculate a derivative value of a function.\n\n    >>> def f(x):\n    ...     return 2 * x**2 + 3\n    >>> x = 3.0 # derivative point\n    >>> h = 0.1 # differential step\n    >>> Np = 3 # point number for central derivative\n    >>> weights = _central_diff_weights(Np) # weights for first derivative\n    >>> vals = [f(x + (i - Np/2) * h) for i in range(Np)]\n    >>> sum(w * v for (w, v) in zip(weights, vals))/h\n    11.79999999999998\n\n    This value is close to the analytical solution:\n    f'(x) = 4x, so f'(3) = 12\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Finite_difference\n\n    "
    if Np < ndiv + 1:
        raise ValueError('Number of points must be at least the derivative order + 1.')
    if Np % 2 == 0:
        raise ValueError('The number of points must be odd.')
    from scipy import linalg
    ho = Np >> 1
    x = arange(-ho, ho + 1.0)
    x = x[:, newaxis]
    X = x ** 0.0
    for k in range(1, Np):
        X = hstack([X, x ** k])
    w = prod(arange(1, ndiv + 1), axis=0) * linalg.inv(X)[ndiv]
    return w

def _derivative(func, x0, dx=1.0, n=1, args=(), order=3):
    if False:
        print('Hello World!')
    '\n    Find the nth derivative of a function at a point.\n\n    Given a function, use a central difference formula with spacing `dx` to\n    compute the nth derivative at `x0`.\n\n    Parameters\n    ----------\n    func : function\n        Input function.\n    x0 : float\n        The point at which the nth derivative is found.\n    dx : float, optional\n        Spacing.\n    n : int, optional\n        Order of the derivative. Default is 1.\n    args : tuple, optional\n        Arguments\n    order : int, optional\n        Number of points to use, must be odd.\n\n    Notes\n    -----\n    Decreasing the step size too small can result in round-off error.\n\n    Examples\n    --------\n    >>> def f(x):\n    ...     return x**3 + x**2\n    >>> _derivative(f, 1.0, dx=1e-6)\n    4.9999999999217337\n\n    '
    if order < n + 1:
        raise ValueError("'order' (the number of points used to compute the derivative), must be at least the derivative order 'n' + 1.")
    if order % 2 == 0:
        raise ValueError("'order' (the number of points used to compute the derivative) must be odd.")
    if n == 1:
        if order == 3:
            weights = array([-1, 0, 1]) / 2.0
        elif order == 5:
            weights = array([1, -8, 0, 8, -1]) / 12.0
        elif order == 7:
            weights = array([-1, 9, -45, 0, 45, -9, 1]) / 60.0
        elif order == 9:
            weights = array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0
        else:
            weights = _central_diff_weights(order, 1)
    elif n == 2:
        if order == 3:
            weights = array([1, -2.0, 1])
        elif order == 5:
            weights = array([-1, 16, -30, 16, -1]) / 12.0
        elif order == 7:
            weights = array([2, -27, 270, -490, 270, -27, 2]) / 180.0
        elif order == 9:
            weights = array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]) / 5040.0
        else:
            weights = _central_diff_weights(order, 2)
    else:
        weights = _central_diff_weights(order, n)
    val = 0.0
    ho = order >> 1
    for k in range(order):
        val += weights[k] * func(x0 + (k - ho) * dx, *args)
    return val / prod((dx,) * n, axis=0)