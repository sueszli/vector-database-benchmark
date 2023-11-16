import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial

def _isscalar(x):
    if False:
        print('Hello World!')
    'Check whether x is if a scalar type, or 0-dim'
    return cupy.isscalar(x) or (hasattr(x, 'shape') and x.shape == ())

class _Interpolator1D:
    """Common features in univariate interpolation.

    Deal with input data type and interpolation axis rolling. The
    actual interpolator can assume the y-data is of shape (n, r) where
    `n` is the number of x-points, and `r` the number of variables,
    and use self.dtype as the y-data type.

    Attributes
    ----------
    _y_axis : Axis along which the interpolation goes in the
        original array
    _y_extra_shape : Additional shape of the input arrays, excluding
        the interpolation axis
    dtype : Dtype of the y-data arrays. It can be set via _set_dtype,
        which forces it to be float or complex

    Methods
    -------
    __call__
    _prepare_x
    _finish_y
    _reshape_y
    _reshape_yi
    _set_yi
    _set_dtype
    _evaluate

    """

    def __init__(self, xi=None, yi=None, axis=None):
        if False:
            return 10
        self._y_axis = axis
        self._y_extra_shape = None
        self.dtype = None
        if yi is not None:
            self._set_yi(yi, xi=xi, axis=axis)

    def __call__(self, x):
        if False:
            while True:
                i = 10
        'Evaluate the interpolant\n\n        Parameters\n        ----------\n        x : cupy.ndarray\n            The points to evaluate the interpolant\n\n        Returns\n        -------\n        y : cupy.ndarray\n            Interpolated values. Shape is determined by replacing\n            the interpolation axis in the original array with the shape of x\n\n        Notes\n        -----\n        Input values `x` must be convertible to `float` values like `int`\n        or `float`.\n\n        '
        (x, x_shape) = self._prepare_x(x)
        y = self._evaluate(x)
        return self._finish_y(y, x_shape)

    def _evaluate(self, x):
        if False:
            while True:
                i = 10
        '\n        Actually evaluate the value of the interpolator\n        '
        raise NotImplementedError()

    def _prepare_x(self, x):
        if False:
            while True:
                i = 10
        '\n        Reshape input array to 1-D\n        '
        x = _asarray_validated(x, check_finite=False, as_inexact=True)
        x_shape = x.shape
        return (x.ravel(), x_shape)

    def _finish_y(self, y, x_shape):
        if False:
            while True:
                i = 10
        '\n        Reshape interpolated y back to an N-D array similar to initial y\n        '
        y = y.reshape(x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = list(range(nx, nx + self._y_axis)) + list(range(nx)) + list(range(nx + self._y_axis, nx + ny))
            y = y.transpose(s)
        return y

    def _reshape_yi(self, yi, check=False):
        if False:
            print('Hello World!')
        '\n        Reshape the updated yi to a 1-D array\n        '
        yi = cupy.moveaxis(yi, self._y_axis, 0)
        if check and yi.shape[1:] != self._y_extra_shape:
            ok_shape = '%r + (N,) + %r' % (self._y_extra_shape[-self._y_axis:], self._y_extra_shape[:-self._y_axis])
            raise ValueError('Data must be of shape %s' % ok_shape)
        return yi.reshape((yi.shape[0], -1))

    def _set_yi(self, yi, xi=None, axis=None):
        if False:
            for i in range(10):
                print('nop')
        if axis is None:
            axis = self._y_axis
        if axis is None:
            raise ValueError('no interpolation axis specified')
        shape = yi.shape
        if shape == ():
            shape = (1,)
        if xi is not None and shape[axis] != len(xi):
            raise ValueError('x and y arrays must be equal in length along interpolation axis.')
        self._y_axis = axis % yi.ndim
        self._y_extra_shape = yi.shape[:self._y_axis] + yi.shape[self._y_axis + 1:]
        self.dtype = None
        self._set_dtype(yi.dtype)

    def _set_dtype(self, dtype, union=False):
        if False:
            for i in range(10):
                print('nop')
        if cupy.issubdtype(dtype, cupy.complexfloating) or cupy.issubdtype(self.dtype, cupy.complexfloating):
            self.dtype = cupy.complex_
        elif not union or self.dtype != cupy.complex_:
            self.dtype = cupy.float_

class _Interpolator1DWithDerivatives(_Interpolator1D):

    def derivatives(self, x, der=None):
        if False:
            while True:
                i = 10
        'Evaluate many derivatives of the polynomial at the point x.\n\n        The function produce an array of all derivative values at\n        the point x.\n\n        Parameters\n        ----------\n        x : cupy.ndarray\n            Point or points at which to evaluate the derivatives\n        der : int or None, optional\n            How many derivatives to extract; None for all potentially\n            nonzero derivatives (that is a number equal to the number\n            of points). This number includes the function value as 0th\n            derivative\n\n        Returns\n        -------\n        d : cupy.ndarray\n            Array with derivatives; d[j] contains the jth derivative.\n            Shape of d[j] is determined by replacing the interpolation\n            axis in the original array with the shape of x\n\n        '
        (x, x_shape) = self._prepare_x(x)
        y = self._evaluate_derivatives(x, der)
        y = y.reshape((y.shape[0],) + x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = [0] + list(range(nx + 1, nx + self._y_axis + 1)) + list(range(1, nx + 1)) + list(range(nx + 1 + self._y_axis, nx + ny + 1))
            y = y.transpose(s)
        return y

    def derivative(self, x, der=1):
        if False:
            print('Hello World!')
        'Evaluate one derivative of the polynomial at the point x\n\n        Parameters\n        ----------\n        x : cupy.ndarray\n            Point or points at which to evaluate the derivatives\n        der : integer, optional\n            Which derivative to extract. This number includes the\n            function value as 0th derivative\n\n        Returns\n        -------\n        d : cupy.ndarray\n            Derivative interpolated at the x-points. Shape of d is\n            determined by replacing the interpolation axis in the\n            original array with the shape of x\n\n        Notes\n        -----\n        This is computed by evaluating all derivatives up to the desired\n        one (using self.derivatives()) and then discarding the rest.\n\n        '
        (x, x_shape) = self._prepare_x(x)
        y = self._evaluate_derivatives(x, der + 1)
        return self._finish_y(y[der], x_shape)

class BarycentricInterpolator(_Interpolator1D):
    """The interpolating polynomial for a set of points.

    Constructs a polynomial that passes through a given set of points.
    Allows evaluation of the polynomial, efficient changing of the y
    values to be interpolated, and updating by adding more x values.
    For reasons of numerical stability, this function does not compute
    the coefficients of the polynomial.
    The value `yi` need to be provided before the function is
    evaluated, but none of the preprocessing depends on them,
    so rapid updates are possible.

    Parameters
    ----------
    xi : cupy.ndarray
        1-D array of x-coordinates of the points the polynomial should
        pass through
    yi : cupy.ndarray, optional
        The y-coordinates of the points the polynomial should pass through.
        If None, the y values will be supplied later via the `set_y` method
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values

    See Also
    --------
    scipy.interpolate.BarycentricInterpolator

    """

    def __init__(self, xi, yi=None, axis=0):
        if False:
            return 10
        _Interpolator1D.__init__(self, xi, yi, axis)
        self.xi = xi.astype(cupy.float_)
        self.set_yi(yi)
        self.n = len(self.xi)
        self._inv_capacity = 4.0 / (cupy.max(self.xi) - cupy.min(self.xi))
        permute = cupy.random.permutation(self.n)
        inv_permute = cupy.zeros(self.n, dtype=cupy.int32)
        inv_permute[permute] = cupy.arange(self.n)
        self.wi = cupy.zeros(self.n)
        for i in range(self.n):
            dist = self._inv_capacity * (self.xi[i] - self.xi[permute])
            dist[inv_permute[i]] = 1.0
            self.wi[i] = 1.0 / cupy.prod(dist)

    def set_yi(self, yi, axis=None):
        if False:
            i = 10
            return i + 15
        'Update the y values to be interpolated.\n\n        The barycentric interpolation algorithm requires the calculation\n        of weights, but these depend only on the xi. The yi can be changed\n        at any time.\n\n        Parameters\n        ----------\n        yi : cupy.ndarray\n            The y-coordinates of the points the polynomial should pass\n            through. If None, the y values will be supplied later.\n        axis : int, optional\n            Axis in the yi array corresponding to the x-coordinate values\n\n        '
        if yi is None:
            self.yi = None
            return
        self._set_yi(yi, xi=self.xi, axis=axis)
        self.yi = self._reshape_yi(yi)
        (self.n, self.r) = self.yi.shape

    def add_xi(self, xi, yi=None):
        if False:
            return 10
        'Add more x values to the set to be interpolated.\n\n        The barycentric interpolation algorithm allows easy updating\n        by adding more points for the polynomial to pass through.\n\n        Parameters\n        ----------\n        xi : cupy.ndarray\n            The x-coordinates of the points that the polynomial should\n            pass through\n        yi : cupy.ndarray, optional\n            The y-coordinates of the points the polynomial should pass\n            through. Should have shape ``(xi.size, R)``; if R > 1 then\n            the polynomial is vector-valued\n            If `yi` is not given, the y values will be supplied later.\n            `yi` should be given if and only if the interpolator has y\n            values specified\n\n        '
        if yi is not None:
            if self.yi is None:
                raise ValueError('No previous yi value to update!')
            yi = self._reshape_yi(yi, check=True)
            self.yi = cupy.vstack((self.yi, yi))
        elif self.yi is not None:
            raise ValueError('No update to yi provided!')
        old_n = self.n
        self.xi = cupy.concatenate((self.xi, xi))
        self.n = len(self.xi)
        self.wi **= -1
        old_wi = self.wi
        self.wi = cupy.zeros(self.n)
        self.wi[:old_n] = old_wi
        for j in range(old_n, self.n):
            self.wi[:j] *= self._inv_capacity * (self.xi[j] - self.xi[:j])
            self.wi[j] = cupy.prod(self._inv_capacity * (self.xi[:j] - self.xi[j]))
        self.wi **= -1

    def __call__(self, x):
        if False:
            i = 10
            return i + 15
        'Evaluate the interpolating polynomial at the points x.\n\n        Parameters\n        ----------\n        x : cupy.ndarray\n            Points to evaluate the interpolant at\n\n        Returns\n        -------\n        y : cupy.ndarray\n            Interpolated values. Shape is determined by replacing the\n            interpolation axis in the original array with the shape of x\n\n        Notes\n        -----\n        Currently the code computes an outer product between x and the\n        weights, that is, it constructs an intermediate array of size\n        N by len(x), where N is the degree of the polynomial.\n\n        '
        return super().__call__(x)

    def _evaluate(self, x):
        if False:
            while True:
                i = 10
        if x.size == 0:
            p = cupy.zeros((0, self.r), dtype=self.dtype)
        else:
            c = x[..., cupy.newaxis] - self.xi
            z = c == 0
            c[z] = 1
            c = self.wi / c
            p = cupy.dot(c, self.yi) / cupy.sum(c, axis=-1)[..., cupy.newaxis]
            r = cupy.nonzero(z)
            if len(r) == 1:
                if len(r[0]) > 0:
                    p = self.yi[r[0][0]]
            else:
                p[r[:-1]] = self.yi[r[-1]]
        return p

def barycentric_interpolate(xi, yi, x, axis=0):
    if False:
        while True:
            i = 10
    'Convenience function for polynomial interpolation.\n\n    Constructs a polynomial that passes through a given\n    set of points, then evaluates the polynomial. For\n    reasons of numerical stability, this function does\n    not compute the coefficients of the polynomial.\n\n    Parameters\n    ----------\n    xi : cupy.ndarray\n        1-D array of coordinates of the points the polynomial\n        should pass through\n    yi : cupy.ndarray\n        y-coordinates of the points the polynomial should pass\n        through\n    x : scalar or cupy.ndarray\n        Points to evaluate the interpolator at\n    axis : int, optional\n        Axis in the yi array corresponding to the x-coordinate\n        values\n\n    Returns\n    -------\n    y : scalar or cupy.ndarray\n        Interpolated values. Shape is determined by replacing\n        the interpolation axis in the original array with the\n        shape x\n\n    See Also\n    --------\n    scipy.interpolate.barycentric_interpolate\n\n    '
    return BarycentricInterpolator(xi, yi, axis=axis)(x)

class KroghInterpolator(_Interpolator1DWithDerivatives):
    """Interpolating polynomial for a set of points.

    The polynomial passes through all the pairs (xi,yi). One may
    additionally specify a number of derivatives at each point xi;
    this is done by repeating the value xi and specifying the
    derivatives as successive yi values
    Allows evaluation of the polynomial and all its derivatives.
    For reasons of numerical stability, this function does not compute
    the coefficients of the polynomial, although they can be obtained
    by evaluating all the derivatives.

    Parameters
    ----------
    xi : cupy.ndarray, length N
        x-coordinate, must be sorted in increasing order
    yi : cupy.ndarray
        y-coordinate, when a xi occurs two or more times in a row,
        the corresponding yi's represent derivative values
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    """

    def __init__(self, xi, yi, axis=0):
        if False:
            for i in range(10):
                print('nop')
        _Interpolator1DWithDerivatives.__init__(self, xi, yi, axis)
        self.xi = xi.astype(cupy.float_)
        self.yi = self._reshape_yi(yi)
        (self.n, self.r) = self.yi.shape
        c = cupy.zeros((self.n + 1, self.r), dtype=self.dtype)
        c[0] = self.yi[0]
        Vk = cupy.zeros((self.n, self.r), dtype=self.dtype)
        for k in range(1, self.n):
            s = 0
            while s <= k and xi[k - s] == xi[k]:
                s += 1
            s -= 1
            Vk[0] = self.yi[k] / float_factorial(s)
            for i in range(k - s):
                if xi[i] == xi[k]:
                    raise ValueError("Elements if `xi` can't be equal.")
                if s == 0:
                    Vk[i + 1] = (c[i] - Vk[i]) / (xi[i] - xi[k])
                else:
                    Vk[i + 1] = (Vk[i + 1] - Vk[i]) / (xi[i] - xi[k])
            c[k] = Vk[k - s]
        self.c = c

    def _evaluate(self, x):
        if False:
            print('Hello World!')
        pi = 1
        p = cupy.zeros((len(x), self.r), dtype=self.dtype)
        p += self.c[0, cupy.newaxis, :]
        for k in range(1, self.n):
            w = x - self.xi[k - 1]
            pi = w * pi
            p += pi[:, cupy.newaxis] * self.c[k]
        return p

    def _evaluate_derivatives(self, x, der=None):
        if False:
            i = 10
            return i + 15
        n = self.n
        r = self.r
        if der is None:
            der = self.n
        pi = cupy.zeros((n, len(x)))
        w = cupy.zeros((n, len(x)))
        pi[0] = 1
        p = cupy.zeros((len(x), self.r), dtype=self.dtype)
        p += self.c[0, cupy.newaxis, :]
        for k in range(1, n):
            w[k - 1] = x - self.xi[k - 1]
            pi[k] = w[k - 1] * pi[k - 1]
            p += pi[k, :, cupy.newaxis] * self.c[k]
        cn = cupy.zeros((max(der, n + 1), len(x), r), dtype=self.dtype)
        cn[:n + 1, :, :] += self.c[:n + 1, cupy.newaxis, :]
        cn[0] = p
        for k in range(1, n):
            for i in range(1, n - k + 1):
                pi[i] = w[k + i - 1] * pi[i - 1] + pi[i]
                cn[k] = cn[k] + pi[i, :, cupy.newaxis] * cn[k + i]
            cn[k] *= float_factorial(k)
        cn[n, :, :] = 0
        return cn[:der]

def krogh_interpolate(xi, yi, x, der=0, axis=0):
    if False:
        i = 10
        return i + 15
    "Convenience function for polynomial interpolation\n\n    Parameters\n    ----------\n    xi : cupy.ndarray\n        x-coordinate\n    yi : cupy.ndarray\n        y-coordinates, of shape ``(xi.size, R)``. Interpreted as\n        vectors of length R, or scalars if R=1\n    x : cupy.ndarray\n        Point or points at which to evaluate the derivatives\n    der : int or list, optional\n        How many derivatives to extract; None for all potentially\n        nonzero derivatives (that is a number equal to the number\n        of points), or a list of derivatives to extract. This number\n        includes the function value as 0th derivative\n    axis : int, optional\n        Axis in the yi array corresponding to the x-coordinate values\n\n    Returns\n    -------\n    d : cupy.ndarray\n        If the interpolator's values are R-D then the\n        returned array will be the number of derivatives by N by R.\n        If `x` is a scalar, the middle dimension will be dropped; if\n        the `yi` are scalars then the last dimension will be dropped\n\n    See Also\n    --------\n    scipy.interpolate.krogh_interpolate\n\n    "
    P = KroghInterpolator(xi, yi, axis=axis)
    if der == 0:
        return P(x)
    elif _isscalar(der):
        return P.derivative(x, der=der)
    else:
        return P.derivatives(x, der=cupy.amax(der) + 1)[der]