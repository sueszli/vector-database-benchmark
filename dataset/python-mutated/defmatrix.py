__all__ = ['matrix', 'bmat', 'asmatrix']
import sys
import warnings
import ast
from .._utils import set_module
import numpy._core.numeric as N
from numpy._core.numeric import concatenate, isscalar
from numpy.linalg import matrix_power

def _convert_from_string(data):
    if False:
        for i in range(10):
            print('nop')
    for char in '[]':
        data = data.replace(char, '')
    rows = data.split(';')
    newdata = []
    count = 0
    for row in rows:
        trow = row.split(',')
        newrow = []
        for col in trow:
            temp = col.split()
            newrow.extend(map(ast.literal_eval, temp))
        if count == 0:
            Ncols = len(newrow)
        elif len(newrow) != Ncols:
            raise ValueError('Rows not the same size.')
        count += 1
        newdata.append(newrow)
    return newdata

@set_module('numpy')
def asmatrix(data, dtype=None):
    if False:
        while True:
            i = 10
    '\n    Interpret the input as a matrix.\n\n    Unlike `matrix`, `asmatrix` does not make a copy if the input is already\n    a matrix or an ndarray.  Equivalent to ``matrix(data, copy=False)``.\n\n    Parameters\n    ----------\n    data : array_like\n        Input data.\n    dtype : data-type\n       Data-type of the output matrix.\n\n    Returns\n    -------\n    mat : matrix\n        `data` interpreted as a matrix.\n\n    Examples\n    --------\n    >>> x = np.array([[1, 2], [3, 4]])\n\n    >>> m = np.asmatrix(x)\n\n    >>> x[0,0] = 5\n\n    >>> m\n    matrix([[5, 2],\n            [3, 4]])\n\n    '
    return matrix(data, dtype=dtype, copy=False)

@set_module('numpy')
class matrix(N.ndarray):
    """
    matrix(data, dtype=None, copy=True)

    Returns a matrix from an array-like object, or from a string of data.

    A matrix is a specialized 2-D array that retains its 2-D nature
    through operations.  It has certain special operators, such as ``*``
    (matrix multiplication) and ``**`` (matrix power).

    .. note:: It is no longer recommended to use this class, even for linear
              algebra. Instead use regular arrays. The class may be removed
              in the future.

    Parameters
    ----------
    data : array_like or string
       If `data` is a string, it is interpreted as a matrix with commas
       or spaces separating columns, and semicolons separating rows.
    dtype : data-type
       Data-type of the output matrix.
    copy : bool
       If `data` is already an `ndarray`, then this flag determines
       whether the data is copied (the default), or whether a view is
       constructed.

    See Also
    --------
    array

    Examples
    --------
    >>> a = np.matrix('1 2; 3 4')
    >>> a
    matrix([[1, 2],
            [3, 4]])

    >>> np.matrix([[1, 2], [3, 4]])
    matrix([[1, 2],
            [3, 4]])

    """
    __array_priority__ = 10.0

    def __new__(subtype, data, dtype=None, copy=True):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('the matrix subclass is not the recommended way to represent matrices or deal with linear algebra (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Please adjust your code to use regular ndarray.', PendingDeprecationWarning, stacklevel=2)
        if isinstance(data, matrix):
            dtype2 = data.dtype
            if dtype is None:
                dtype = dtype2
            if dtype2 == dtype and (not copy):
                return data
            return data.astype(dtype)
        if isinstance(data, N.ndarray):
            if dtype is None:
                intype = data.dtype
            else:
                intype = N.dtype(dtype)
            new = data.view(subtype)
            if intype != data.dtype:
                return new.astype(intype)
            if copy:
                return new.copy()
            else:
                return new
        if isinstance(data, str):
            data = _convert_from_string(data)
        arr = N.array(data, dtype=dtype, copy=copy)
        ndim = arr.ndim
        shape = arr.shape
        if ndim > 2:
            raise ValueError('matrix must be 2-dimensional')
        elif ndim == 0:
            shape = (1, 1)
        elif ndim == 1:
            shape = (1, shape[0])
        order = 'C'
        if ndim == 2 and arr.flags.fortran:
            order = 'F'
        if not (order or arr.flags.contiguous):
            arr = arr.copy()
        ret = N.ndarray.__new__(subtype, shape, arr.dtype, buffer=arr, order=order)
        return ret

    def __array_finalize__(self, obj):
        if False:
            for i in range(10):
                print('nop')
        self._getitem = False
        if isinstance(obj, matrix) and obj._getitem:
            return
        ndim = self.ndim
        if ndim == 2:
            return
        if ndim > 2:
            newshape = tuple([x for x in self.shape if x > 1])
            ndim = len(newshape)
            if ndim == 2:
                self.shape = newshape
                return
            elif ndim > 2:
                raise ValueError('shape too large to be a matrix.')
        else:
            newshape = self.shape
        if ndim == 0:
            self.shape = (1, 1)
        elif ndim == 1:
            self.shape = (1, newshape[0])
        return

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        self._getitem = True
        try:
            out = N.ndarray.__getitem__(self, index)
        finally:
            self._getitem = False
        if not isinstance(out, N.ndarray):
            return out
        if out.ndim == 0:
            return out[()]
        if out.ndim == 1:
            sh = out.shape[0]
            try:
                n = len(index)
            except Exception:
                n = 0
            if n > 1 and isscalar(index[1]):
                out.shape = (sh, 1)
            else:
                out.shape = (1, sh)
        return out

    def __mul__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, (N.ndarray, list, tuple)):
            return N.dot(self, asmatrix(other))
        if isscalar(other) or not hasattr(other, '__rmul__'):
            return N.dot(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if False:
            print('Hello World!')
        return N.dot(other, self)

    def __imul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        self[:] = self * other
        return self

    def __pow__(self, other):
        if False:
            print('Hello World!')
        return matrix_power(self, other)

    def __ipow__(self, other):
        if False:
            i = 10
            return i + 15
        self[:] = self ** other
        return self

    def __rpow__(self, other):
        if False:
            i = 10
            return i + 15
        return NotImplemented

    def _align(self, axis):
        if False:
            print('Hello World!')
        'A convenience function for operations that need to preserve axis\n        orientation.\n        '
        if axis is None:
            return self[0, 0]
        elif axis == 0:
            return self
        elif axis == 1:
            return self.transpose()
        else:
            raise ValueError('unsupported axis')

    def _collapse(self, axis):
        if False:
            while True:
                i = 10
        'A convenience function for operations that want to collapse\n        to a scalar like _align, but are using keepdims=True\n        '
        if axis is None:
            return self[0, 0]
        else:
            return self

    def tolist(self):
        if False:
            print('Hello World!')
        '\n        Return the matrix as a (possibly nested) list.\n\n        See `ndarray.tolist` for full documentation.\n\n        See Also\n        --------\n        ndarray.tolist\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.tolist()\n        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]\n\n        '
        return self.__array__().tolist()

    def sum(self, axis=None, dtype=None, out=None):
        if False:
            print('Hello World!')
        "\n        Returns the sum of the matrix elements, along the given axis.\n\n        Refer to `numpy.sum` for full documentation.\n\n        See Also\n        --------\n        numpy.sum\n\n        Notes\n        -----\n        This is the same as `ndarray.sum`, except that where an `ndarray` would\n        be returned, a `matrix` object is returned instead.\n\n        Examples\n        --------\n        >>> x = np.matrix([[1, 2], [4, 3]])\n        >>> x.sum()\n        10\n        >>> x.sum(axis=1)\n        matrix([[3],\n                [7]])\n        >>> x.sum(axis=1, dtype='float')\n        matrix([[3.],\n                [7.]])\n        >>> out = np.zeros((2, 1), dtype='float')\n        >>> x.sum(axis=1, dtype='float', out=np.asmatrix(out))\n        matrix([[3.],\n                [7.]])\n\n        "
        return N.ndarray.sum(self, axis, dtype, out, keepdims=True)._collapse(axis)

    def squeeze(self, axis=None):
        if False:
            print('Hello World!')
        '\n        Return a possibly reshaped matrix.\n\n        Refer to `numpy.squeeze` for more documentation.\n\n        Parameters\n        ----------\n        axis : None or int or tuple of ints, optional\n            Selects a subset of the axes of length one in the shape.\n            If an axis is selected with shape entry greater than one,\n            an error is raised.\n\n        Returns\n        -------\n        squeezed : matrix\n            The matrix, but as a (1, N) matrix if it had shape (N, 1).\n\n        See Also\n        --------\n        numpy.squeeze : related function\n\n        Notes\n        -----\n        If `m` has a single column then that column is returned\n        as the single row of a matrix.  Otherwise `m` is returned.\n        The returned matrix is always either `m` itself or a view into `m`.\n        Supplying an axis keyword argument will not affect the returned matrix\n        but it may cause an error to be raised.\n\n        Examples\n        --------\n        >>> c = np.matrix([[1], [2]])\n        >>> c\n        matrix([[1],\n                [2]])\n        >>> c.squeeze()\n        matrix([[1, 2]])\n        >>> r = c.T\n        >>> r\n        matrix([[1, 2]])\n        >>> r.squeeze()\n        matrix([[1, 2]])\n        >>> m = np.matrix([[1, 2], [3, 4]])\n        >>> m.squeeze()\n        matrix([[1, 2],\n                [3, 4]])\n\n        '
        return N.ndarray.squeeze(self, axis=axis)

    def flatten(self, order='C'):
        if False:
            return 10
        "\n        Return a flattened copy of the matrix.\n\n        All `N` elements of the matrix are placed into a single row.\n\n        Parameters\n        ----------\n        order : {'C', 'F', 'A', 'K'}, optional\n            'C' means to flatten in row-major (C-style) order. 'F' means to\n            flatten in column-major (Fortran-style) order. 'A' means to\n            flatten in column-major order if `m` is Fortran *contiguous* in\n            memory, row-major order otherwise. 'K' means to flatten `m` in\n            the order the elements occur in memory. The default is 'C'.\n\n        Returns\n        -------\n        y : matrix\n            A copy of the matrix, flattened to a `(1, N)` matrix where `N`\n            is the number of elements in the original matrix.\n\n        See Also\n        --------\n        ravel : Return a flattened array.\n        flat : A 1-D flat iterator over the matrix.\n\n        Examples\n        --------\n        >>> m = np.matrix([[1,2], [3,4]])\n        >>> m.flatten()\n        matrix([[1, 2, 3, 4]])\n        >>> m.flatten('F')\n        matrix([[1, 3, 2, 4]])\n\n        "
        return N.ndarray.flatten(self, order=order)

    def mean(self, axis=None, dtype=None, out=None):
        if False:
            return 10
        '\n        Returns the average of the matrix elements along the given axis.\n\n        Refer to `numpy.mean` for full documentation.\n\n        See Also\n        --------\n        numpy.mean\n\n        Notes\n        -----\n        Same as `ndarray.mean` except that, where that returns an `ndarray`,\n        this returns a `matrix` object.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3, 4)))\n        >>> x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.mean()\n        5.5\n        >>> x.mean(0)\n        matrix([[4., 5., 6., 7.]])\n        >>> x.mean(1)\n        matrix([[ 1.5],\n                [ 5.5],\n                [ 9.5]])\n\n        '
        return N.ndarray.mean(self, axis, dtype, out, keepdims=True)._collapse(axis)

    def std(self, axis=None, dtype=None, out=None, ddof=0):
        if False:
            return 10
        '\n        Return the standard deviation of the array elements along the given axis.\n\n        Refer to `numpy.std` for full documentation.\n\n        See Also\n        --------\n        numpy.std\n\n        Notes\n        -----\n        This is the same as `ndarray.std`, except that where an `ndarray` would\n        be returned, a `matrix` object is returned instead.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3, 4)))\n        >>> x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.std()\n        3.4520525295346629 # may vary\n        >>> x.std(0)\n        matrix([[ 3.26598632,  3.26598632,  3.26598632,  3.26598632]]) # may vary\n        >>> x.std(1)\n        matrix([[ 1.11803399],\n                [ 1.11803399],\n                [ 1.11803399]])\n\n        '
        return N.ndarray.std(self, axis, dtype, out, ddof, keepdims=True)._collapse(axis)

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        if False:
            while True:
                i = 10
        '\n        Returns the variance of the matrix elements, along the given axis.\n\n        Refer to `numpy.var` for full documentation.\n\n        See Also\n        --------\n        numpy.var\n\n        Notes\n        -----\n        This is the same as `ndarray.var`, except that where an `ndarray` would\n        be returned, a `matrix` object is returned instead.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3, 4)))\n        >>> x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.var()\n        11.916666666666666\n        >>> x.var(0)\n        matrix([[ 10.66666667,  10.66666667,  10.66666667,  10.66666667]]) # may vary\n        >>> x.var(1)\n        matrix([[1.25],\n                [1.25],\n                [1.25]])\n\n        '
        return N.ndarray.var(self, axis, dtype, out, ddof, keepdims=True)._collapse(axis)

    def prod(self, axis=None, dtype=None, out=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the product of the array elements over the given axis.\n\n        Refer to `prod` for full documentation.\n\n        See Also\n        --------\n        prod, ndarray.prod\n\n        Notes\n        -----\n        Same as `ndarray.prod`, except, where that returns an `ndarray`, this\n        returns a `matrix` object instead.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.prod()\n        0\n        >>> x.prod(0)\n        matrix([[  0,  45, 120, 231]])\n        >>> x.prod(1)\n        matrix([[   0],\n                [ 840],\n                [7920]])\n\n        '
        return N.ndarray.prod(self, axis, dtype, out, keepdims=True)._collapse(axis)

    def any(self, axis=None, out=None):
        if False:
            i = 10
            return i + 15
        '\n        Test whether any array element along a given axis evaluates to True.\n\n        Refer to `numpy.any` for full documentation.\n\n        Parameters\n        ----------\n        axis : int, optional\n            Axis along which logical OR is performed\n        out : ndarray, optional\n            Output to existing array instead of creating new one, must have\n            same shape as expected output\n\n        Returns\n        -------\n            any : bool, ndarray\n                Returns a single bool if `axis` is ``None``; otherwise,\n                returns `ndarray`\n\n        '
        return N.ndarray.any(self, axis, out, keepdims=True)._collapse(axis)

    def all(self, axis=None, out=None):
        if False:
            i = 10
            return i + 15
        '\n        Test whether all matrix elements along a given axis evaluate to True.\n\n        Parameters\n        ----------\n        See `numpy.all` for complete descriptions\n\n        See Also\n        --------\n        numpy.all\n\n        Notes\n        -----\n        This is the same as `ndarray.all`, but it returns a `matrix` object.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> y = x[0]; y\n        matrix([[0, 1, 2, 3]])\n        >>> (x == y)\n        matrix([[ True,  True,  True,  True],\n                [False, False, False, False],\n                [False, False, False, False]])\n        >>> (x == y).all()\n        False\n        >>> (x == y).all(0)\n        matrix([[False, False, False, False]])\n        >>> (x == y).all(1)\n        matrix([[ True],\n                [False],\n                [False]])\n\n        '
        return N.ndarray.all(self, axis, out, keepdims=True)._collapse(axis)

    def max(self, axis=None, out=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the maximum value along an axis.\n\n        Parameters\n        ----------\n        See `amax` for complete descriptions\n\n        See Also\n        --------\n        amax, ndarray.max\n\n        Notes\n        -----\n        This is the same as `ndarray.max`, but returns a `matrix` object\n        where `ndarray.max` would return an ndarray.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.max()\n        11\n        >>> x.max(0)\n        matrix([[ 8,  9, 10, 11]])\n        >>> x.max(1)\n        matrix([[ 3],\n                [ 7],\n                [11]])\n\n        '
        return N.ndarray.max(self, axis, out, keepdims=True)._collapse(axis)

    def argmax(self, axis=None, out=None):
        if False:
            print('Hello World!')
        '\n        Indexes of the maximum values along an axis.\n\n        Return the indexes of the first occurrences of the maximum values\n        along the specified axis.  If axis is None, the index is for the\n        flattened matrix.\n\n        Parameters\n        ----------\n        See `numpy.argmax` for complete descriptions\n\n        See Also\n        --------\n        numpy.argmax\n\n        Notes\n        -----\n        This is the same as `ndarray.argmax`, but returns a `matrix` object\n        where `ndarray.argmax` would return an `ndarray`.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.argmax()\n        11\n        >>> x.argmax(0)\n        matrix([[2, 2, 2, 2]])\n        >>> x.argmax(1)\n        matrix([[3],\n                [3],\n                [3]])\n\n        '
        return N.ndarray.argmax(self, axis, out)._align(axis)

    def min(self, axis=None, out=None):
        if False:
            i = 10
            return i + 15
        '\n        Return the minimum value along an axis.\n\n        Parameters\n        ----------\n        See `amin` for complete descriptions.\n\n        See Also\n        --------\n        amin, ndarray.min\n\n        Notes\n        -----\n        This is the same as `ndarray.min`, but returns a `matrix` object\n        where `ndarray.min` would return an ndarray.\n\n        Examples\n        --------\n        >>> x = -np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[  0,  -1,  -2,  -3],\n                [ -4,  -5,  -6,  -7],\n                [ -8,  -9, -10, -11]])\n        >>> x.min()\n        -11\n        >>> x.min(0)\n        matrix([[ -8,  -9, -10, -11]])\n        >>> x.min(1)\n        matrix([[ -3],\n                [ -7],\n                [-11]])\n\n        '
        return N.ndarray.min(self, axis, out, keepdims=True)._collapse(axis)

    def argmin(self, axis=None, out=None):
        if False:
            while True:
                i = 10
        '\n        Indexes of the minimum values along an axis.\n\n        Return the indexes of the first occurrences of the minimum values\n        along the specified axis.  If axis is None, the index is for the\n        flattened matrix.\n\n        Parameters\n        ----------\n        See `numpy.argmin` for complete descriptions.\n\n        See Also\n        --------\n        numpy.argmin\n\n        Notes\n        -----\n        This is the same as `ndarray.argmin`, but returns a `matrix` object\n        where `ndarray.argmin` would return an `ndarray`.\n\n        Examples\n        --------\n        >>> x = -np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[  0,  -1,  -2,  -3],\n                [ -4,  -5,  -6,  -7],\n                [ -8,  -9, -10, -11]])\n        >>> x.argmin()\n        11\n        >>> x.argmin(0)\n        matrix([[2, 2, 2, 2]])\n        >>> x.argmin(1)\n        matrix([[3],\n                [3],\n                [3]])\n\n        '
        return N.ndarray.argmin(self, axis, out)._align(axis)

    def ptp(self, axis=None, out=None):
        if False:
            return 10
        '\n        Peak-to-peak (maximum - minimum) value along the given axis.\n\n        Refer to `numpy.ptp` for full documentation.\n\n        See Also\n        --------\n        numpy.ptp\n\n        Notes\n        -----\n        Same as `ndarray.ptp`, except, where that would return an `ndarray` object,\n        this returns a `matrix` object.\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.ptp()\n        11\n        >>> x.ptp(0)\n        matrix([[8, 8, 8, 8]])\n        >>> x.ptp(1)\n        matrix([[3],\n                [3],\n                [3]])\n\n        '
        return N.ptp(self, axis, out)._align(axis)

    @property
    def I(self):
        if False:
            print('Hello World!')
        "\n        Returns the (multiplicative) inverse of invertible `self`.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        ret : matrix object\n            If `self` is non-singular, `ret` is such that ``ret * self`` ==\n            ``self * ret`` == ``np.matrix(np.eye(self[0,:].size))`` all return\n            ``True``.\n\n        Raises\n        ------\n        numpy.linalg.LinAlgError: Singular matrix\n            If `self` is singular.\n\n        See Also\n        --------\n        linalg.inv\n\n        Examples\n        --------\n        >>> m = np.matrix('[1, 2; 3, 4]'); m\n        matrix([[1, 2],\n                [3, 4]])\n        >>> m.getI()\n        matrix([[-2. ,  1. ],\n                [ 1.5, -0.5]])\n        >>> m.getI() * m\n        matrix([[ 1.,  0.], # may vary\n                [ 0.,  1.]])\n\n        "
        (M, N) = self.shape
        if M == N:
            from numpy.linalg import inv as func
        else:
            from numpy.linalg import pinv as func
        return asmatrix(func(self))

    @property
    def A(self):
        if False:
            i = 10
            return i + 15
        '\n        Return `self` as an `ndarray` object.\n\n        Equivalent to ``np.asarray(self)``.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        ret : ndarray\n            `self` as an `ndarray`\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.getA()\n        array([[ 0,  1,  2,  3],\n               [ 4,  5,  6,  7],\n               [ 8,  9, 10, 11]])\n\n        '
        return self.__array__()

    @property
    def A1(self):
        if False:
            i = 10
            return i + 15
        '\n        Return `self` as a flattened `ndarray`.\n\n        Equivalent to ``np.asarray(x).ravel()``\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        ret : ndarray\n            `self`, 1-D, as an `ndarray`\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4))); x\n        matrix([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n        >>> x.getA1()\n        array([ 0,  1,  2, ...,  9, 10, 11])\n\n\n        '
        return self.__array__().ravel()

    def ravel(self, order='C'):
        if False:
            while True:
                i = 10
        "\n        Return a flattened matrix.\n\n        Refer to `numpy.ravel` for more documentation.\n\n        Parameters\n        ----------\n        order : {'C', 'F', 'A', 'K'}, optional\n            The elements of `m` are read using this index order. 'C' means to\n            index the elements in C-like order, with the last axis index\n            changing fastest, back to the first axis index changing slowest.\n            'F' means to index the elements in Fortran-like index order, with\n            the first index changing fastest, and the last index changing\n            slowest. Note that the 'C' and 'F' options take no account of the\n            memory layout of the underlying array, and only refer to the order\n            of axis indexing.  'A' means to read the elements in Fortran-like\n            index order if `m` is Fortran *contiguous* in memory, C-like order\n            otherwise.  'K' means to read the elements in the order they occur\n            in memory, except for reversing the data when strides are negative.\n            By default, 'C' index order is used.\n\n        Returns\n        -------\n        ret : matrix\n            Return the matrix flattened to shape `(1, N)` where `N`\n            is the number of elements in the original matrix.\n            A copy is made only if necessary.\n\n        See Also\n        --------\n        matrix.flatten : returns a similar output matrix but always a copy\n        matrix.flat : a flat iterator on the array.\n        numpy.ravel : related function which returns an ndarray\n\n        "
        return N.ndarray.ravel(self, order=order)

    @property
    def T(self):
        if False:
            return 10
        "\n        Returns the transpose of the matrix.\n\n        Does *not* conjugate!  For the complex conjugate transpose, use ``.H``.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        ret : matrix object\n            The (non-conjugated) transpose of the matrix.\n\n        See Also\n        --------\n        transpose, getH\n\n        Examples\n        --------\n        >>> m = np.matrix('[1, 2; 3, 4]')\n        >>> m\n        matrix([[1, 2],\n                [3, 4]])\n        >>> m.getT()\n        matrix([[1, 3],\n                [2, 4]])\n\n        "
        return self.transpose()

    @property
    def H(self):
        if False:
            print('Hello World!')
        '\n        Returns the (complex) conjugate transpose of `self`.\n\n        Equivalent to ``np.transpose(self)`` if `self` is real-valued.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        ret : matrix object\n            complex conjugate transpose of `self`\n\n        Examples\n        --------\n        >>> x = np.matrix(np.arange(12).reshape((3,4)))\n        >>> z = x - 1j*x; z\n        matrix([[  0. +0.j,   1. -1.j,   2. -2.j,   3. -3.j],\n                [  4. -4.j,   5. -5.j,   6. -6.j,   7. -7.j],\n                [  8. -8.j,   9. -9.j,  10.-10.j,  11.-11.j]])\n        >>> z.getH()\n        matrix([[ 0. -0.j,  4. +4.j,  8. +8.j],\n                [ 1. +1.j,  5. +5.j,  9. +9.j],\n                [ 2. +2.j,  6. +6.j, 10.+10.j],\n                [ 3. +3.j,  7. +7.j, 11.+11.j]])\n\n        '
        if issubclass(self.dtype.type, N.complexfloating):
            return self.transpose().conjugate()
        else:
            return self.transpose()
    getT = T.fget
    getA = A.fget
    getA1 = A1.fget
    getH = H.fget
    getI = I.fget

def _from_string(str, gdict, ldict):
    if False:
        for i in range(10):
            print('nop')
    rows = str.split(';')
    rowtup = []
    for row in rows:
        trow = row.split(',')
        newrow = []
        for x in trow:
            newrow.extend(x.split())
        trow = newrow
        coltup = []
        for col in trow:
            col = col.strip()
            try:
                thismat = ldict[col]
            except KeyError:
                try:
                    thismat = gdict[col]
                except KeyError as e:
                    raise NameError(f'name {col!r} is not defined') from None
            coltup.append(thismat)
        rowtup.append(concatenate(coltup, axis=-1))
    return concatenate(rowtup, axis=0)

@set_module('numpy')
def bmat(obj, ldict=None, gdict=None):
    if False:
        while True:
            i = 10
    "\n    Build a matrix object from a string, nested sequence, or array.\n\n    Parameters\n    ----------\n    obj : str or array_like\n        Input data. If a string, variables in the current scope may be\n        referenced by name.\n    ldict : dict, optional\n        A dictionary that replaces local operands in current frame.\n        Ignored if `obj` is not a string or `gdict` is None.\n    gdict : dict, optional\n        A dictionary that replaces global operands in current frame.\n        Ignored if `obj` is not a string.\n\n    Returns\n    -------\n    out : matrix\n        Returns a matrix object, which is a specialized 2-D array.\n\n    See Also\n    --------\n    block :\n        A generalization of this function for N-d arrays, that returns normal\n        ndarrays.\n\n    Examples\n    --------\n    >>> A = np.asmatrix('1 1; 1 1')\n    >>> B = np.asmatrix('2 2; 2 2')\n    >>> C = np.asmatrix('3 4; 5 6')\n    >>> D = np.asmatrix('7 8; 9 0')\n\n    All the following expressions construct the same block matrix:\n\n    >>> np.bmat([[A, B], [C, D]])\n    matrix([[1, 1, 2, 2],\n            [1, 1, 2, 2],\n            [3, 4, 7, 8],\n            [5, 6, 9, 0]])\n    >>> np.bmat(np.r_[np.c_[A, B], np.c_[C, D]])\n    matrix([[1, 1, 2, 2],\n            [1, 1, 2, 2],\n            [3, 4, 7, 8],\n            [5, 6, 9, 0]])\n    >>> np.bmat('A,B; C,D')\n    matrix([[1, 1, 2, 2],\n            [1, 1, 2, 2],\n            [3, 4, 7, 8],\n            [5, 6, 9, 0]])\n\n    "
    if isinstance(obj, str):
        if gdict is None:
            frame = sys._getframe().f_back
            glob_dict = frame.f_globals
            loc_dict = frame.f_locals
        else:
            glob_dict = gdict
            loc_dict = ldict
        return matrix(_from_string(obj, glob_dict, loc_dict))
    if isinstance(obj, (tuple, list)):
        arr_rows = []
        for row in obj:
            if isinstance(row, N.ndarray):
                return matrix(concatenate(obj, axis=-1))
            else:
                arr_rows.append(concatenate(row, axis=-1))
        return matrix(concatenate(arr_rows, axis=0))
    if isinstance(obj, N.ndarray):
        return matrix(obj)