"""Interpolation of data points in a plane based on Akima's method.

Akima's interpolation method uses a continuously differentiable sub-spline
built from piecewise cubic polynomials. The resultant curve passes through
the given data points and will appear smooth and natural.

:Author:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2015.01.29

Requirements
------------
* `CPython 2.7 or 3.4 <http://www.python.org>`_
* `Numpy 1.8 <http://www.numpy.org>`_
* `Akima.c 2015.01.29 <http://www.lfd.uci.edu/~gohlke/>`_  (optional speedup)
* `Matplotlib 1.4 <http://www.matplotlib.org>`_  (optional for plotting)

Notes
-----
Consider using `scipy.interpolate.Akima1DInterpolator
<http://docs.scipy.org/doc/scipy/reference/interpolate.html>`_.

References
----------
(1) A new method of interpolation and smooth curve fitting based
    on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4), 589-602.

Examples
--------
>>> def example():
...     '''Plot interpolated Gaussian noise.'''
...     x = numpy.sort(numpy.random.random(10) * 100)
...     y = numpy.random.normal(0.0, 0.1, size=len(x))
...     x2 = numpy.arange(x[0], x[-1], 0.05)
...     y2 = interpolate(x, y, x2)
...     from matplotlib import pyplot
...     pyplot.title("Akima interpolation of Gaussian noise")
...     pyplot.plot(x2, y2, "b-")
...     pyplot.plot(x, y, "ro")
...     pyplot.show()
>>> example()

"""
import numpy
__version__ = '2015.01.29'
__docformat__ = 'restructuredtext en'
__all__ = ('interpolate',)

def interpolate(x, y, x_new, axis=-1, out=None):
    if False:
        while True:
            i = 10
    "Return interpolated data using Akima's method.\n\n    This Python implementation is inspired by the Matlab(r) code by\n    N. Shamsundar. It lacks certain capabilities of the C implementation\n    such as the output array argument and interpolation along an axis of a\n    multidimensional data array.\n\n    Parameters\n    ----------\n    x : array like\n        1D array of monotonically increasing real values.\n    y : array like\n        N-D array of real values. y's length along the interpolation\n        axis must be equal to the length of x.\n    x_new : array like\n        New independent variables.\n    axis : int\n        Specifies axis of y along which to interpolate. Interpolation\n        defaults to last axis of y.\n    out : array\n        Optional array to receive results. Dimension at axis must equal\n        length of x.\n\n    Examples\n    --------\n    >>> interpolate([0, 1, 2], [0, 0, 1], [0.5, 1.5])\n    array([-0.125,  0.375])\n    >>> x = numpy.sort(numpy.random.random(10) * 10)\n    >>> y = numpy.random.normal(0.0, 0.1, size=len(x))\n    >>> z = interpolate(x, y, x)\n    >>> numpy.allclose(y, z)\n    True\n    >>> x = x[:10]\n    >>> y = numpy.reshape(y, (10, -1))\n    >>> z = numpy.reshape(y, (10, -1))\n    >>> interpolate(x, y, x, axis=0, out=z)\n    >>> numpy.allclose(y, z)\n    True\n\n    "
    x = numpy.array(x, dtype=numpy.float64, copy=True)
    y = numpy.array(y, dtype=numpy.float64, copy=True)
    xi = numpy.array(x_new, dtype=numpy.float64, copy=True)
    if axis != -1 or out is not None or y.ndim != 1:
        raise NotImplementedError('implemented in C extension module')
    if x.ndim != 1 or xi.ndim != 1:
        raise ValueError('x-arrays must be one dimensional')
    n = len(x)
    if n < 2:
        raise ValueError('array too small')
    if n != y.shape[axis]:
        raise ValueError('size of x-array must match data shape')
    dx = numpy.diff(x)
    if any(dx <= 0.0):
        raise ValueError('x-axis not valid')
    if any(xi < x[0]) or any(xi > x[-1]):
        raise ValueError('interpolation x-axis out of bounds')
    m = numpy.diff(y) / dx
    mm = 2.0 * m[0] - m[1]
    mmm = 2.0 * mm - m[0]
    mp = 2.0 * m[n - 2] - m[n - 3]
    mpp = 2.0 * mp - m[n - 2]
    m1 = numpy.concatenate(([mmm], [mm], m, [mp], [mpp]))
    dm = numpy.abs(numpy.diff(m1))
    f1 = dm[2:n + 2]
    f2 = dm[0:n]
    f12 = f1 + f2
    ids = numpy.nonzero(f12 > 1e-09 * numpy.max(f12))[0]
    b = m1[1:n + 1]
    b[ids] = (f1[ids] * m1[ids + 1] + f2[ids] * m1[ids + 2]) / f12[ids]
    c = (3.0 * m - 2.0 * b[0:n - 1] - b[1:n]) / dx
    d = (b[0:n - 1] + b[1:n] - 2.0 * m) / dx ** 2
    bins = numpy.digitize(xi, x)
    bins = numpy.minimum(bins, n - 1) - 1
    bb = bins[0:len(xi)]
    wj = xi - x[bb]
    return ((wj * d[bb] + c[bb]) * wj + b[bb]) * wj + y[bb]