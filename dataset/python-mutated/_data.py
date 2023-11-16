import cupy
import numpy as np
from cupy._core import internal
from cupy import _util
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _sputils
_ufuncs = ['arcsin', 'arcsinh', 'arctan', 'arctanh', 'ceil', 'deg2rad', 'expm1', 'floor', 'log1p', 'rad2deg', 'rint', 'sign', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc']

class _data_matrix(_base.spmatrix):

    def __init__(self, data):
        if False:
            return 10
        self.data = data

    @property
    def dtype(self):
        if False:
            for i in range(10):
                print('nop')
        'Data type of the matrix.'
        return self.data.dtype

    def _with_data(self, data, copy=True):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def __abs__(self):
        if False:
            print('Hello World!')
        'Elementwise abosulte.'
        return self._with_data(abs(self.data))

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        'Elementwise negative.'
        return self._with_data(-self.data)

    def astype(self, t):
        if False:
            for i in range(10):
                print('nop')
        'Casts the array to given data type.\n\n        Args:\n            dtype: Type specifier.\n\n        Returns:\n            A copy of the array with a given type.\n\n        '
        return self._with_data(self.data.astype(t))

    def conj(self, copy=True):
        if False:
            while True:
                i = 10
        if cupy.issubdtype(self.dtype, cupy.complexfloating):
            return self._with_data(self.data.conj(), copy=copy)
        elif copy:
            return self.copy()
        else:
            return self
    conj.__doc__ = _base.spmatrix.conj.__doc__

    def copy(self):
        if False:
            return 10
        return self._with_data(self.data.copy(), copy=True)
    copy.__doc__ = _base.spmatrix.copy.__doc__

    def count_nonzero(self):
        if False:
            i = 10
            return i + 15
        'Returns number of non-zero entries.\n\n        .. note::\n           This method counts the actual number of non-zero entories, which\n           does not include explicit zero entries.\n           Instead ``nnz`` returns the number of entries including explicit\n           zeros.\n\n        Returns:\n            Number of non-zero entries.\n\n        '
        return cupy.count_nonzero(self.data)

    def mean(self, axis=None, dtype=None, out=None):
        if False:
            while True:
                i = 10
        'Compute the arithmetic mean along the specified axis.\n\n        Args:\n            axis (int or ``None``): Axis along which the sum is computed.\n                If it is ``None``, it computes the average of all the elements.\n                Select from ``{None, 0, 1, -2, -1}``.\n\n        Returns:\n            cupy.ndarray: Summed array.\n\n        .. seealso::\n           :meth:`scipy.sparse.spmatrix.mean`\n\n        '
        _sputils.validateaxis(axis)
        (nRow, nCol) = self.shape
        data = self.data.copy()
        if axis is None:
            n = nRow * nCol
        elif axis in (0, -2):
            n = nRow
        else:
            n = nCol
        return self._with_data(data / n).sum(axis, dtype, out)

    def power(self, n, dtype=None):
        if False:
            for i in range(10):
                print('nop')
        'Elementwise power function.\n\n        Args:\n            n: Exponent.\n            dtype: Type specifier.\n\n        '
        if dtype is None:
            data = self.data.copy()
        else:
            data = self.data.astype(dtype, copy=True)
        data **= n
        return self._with_data(data)

def _find_missing_index(ind, n):
    if False:
        while True:
            i = 10
    positions = cupy.arange(ind.size)
    diff = ind != positions
    return cupy.where(diff.any(), diff.argmax(), cupy.asarray(ind.size if ind.size < n else -1))

def _non_zero_cmp(mat, am, zero, m):
    if False:
        return 10
    size = np.prod(mat.shape)
    if size == mat.nnz:
        return am
    else:
        ind = mat.row * mat.shape[1] + mat.col
        zero_ind = _find_missing_index(ind, size)
        return cupy.where(m == zero, cupy.minimum(zero_ind, am), zero_ind)

class _minmax_mixin(object):
    """Mixin for min and max methods.
    These are not implemented for dia_matrix, hence the separate class.

    """

    def _min_or_max_axis(self, axis, min_or_max, explicit):
        if False:
            i = 10
            return i + 15
        N = self.shape[axis]
        if N == 0:
            raise ValueError('zero-size array to reduction operation')
        M = self.shape[1 - axis]
        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()
        value = mat._minor_reduce(min_or_max, axis, explicit)
        major_index = cupy.arange(M)
        mask = value != 0
        major_index = cupy.compress(mask, major_index)
        value = cupy.compress(mask, value)
        if axis == 0:
            return _coo.coo_matrix((value, (cupy.zeros(len(value)), major_index)), dtype=self.dtype, shape=(1, M))
        else:
            return _coo.coo_matrix((value, (major_index, cupy.zeros(len(value)))), dtype=self.dtype, shape=(M, 1))

    def _min_or_max(self, axis, out, min_or_max, explicit):
        if False:
            i = 10
            return i + 15
        if out is not None:
            raise ValueError("Sparse matrices do not support an 'out' parameter.")
        _sputils.validateaxis(axis)
        if axis is None:
            if 0 in self.shape:
                raise ValueError('zero-size array to reduction operation')
            zero = cupy.zeros((), dtype=self.dtype)
            if self.nnz == 0:
                return zero
            self.sum_duplicates()
            m = min_or_max(self.data)
            if explicit:
                return m
            if self.nnz != internal.prod(self.shape):
                if min_or_max is cupy.min:
                    m = cupy.minimum(zero, m)
                elif min_or_max is cupy.max:
                    m = cupy.maximum(zero, m)
                else:
                    assert False
            return m
        if axis < 0:
            axis += 2
        return self._min_or_max_axis(axis, min_or_max, explicit)

    def _arg_min_or_max_axis(self, axis, op):
        if False:
            i = 10
            return i + 15
        if self.shape[axis] == 0:
            raise ValueError("Can't apply the operation along a zero-sized dimension.")
        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()
        value = mat._arg_minor_reduce(op, axis)
        if axis == 0:
            return value[None, :]
        else:
            return value[:, None]

    def _arg_min_or_max(self, axis, out, op, compare):
        if False:
            return 10
        if out is not None:
            raise ValueError("Sparse matrices do not support an 'out' parameter.")
        _sputils.validateaxis(axis)
        if axis is None:
            if 0 in self.shape:
                raise ValueError("Can't apply the operation to an empty matrix.")
            if self.nnz == 0:
                return 0
            else:
                zero = cupy.asarray(self.dtype.type(0))
                mat = self.tocoo()
                mat.sum_duplicates()
                am = op(mat.data)
                m = mat.data[am]
                return cupy.where(compare(m, zero), mat.row[am] * mat.shape[1] + mat.col[am], _non_zero_cmp(mat, am, zero, m))
        if axis < 0:
            axis += 2
        return self._arg_min_or_max_axis(axis, op)

    def max(self, axis=None, out=None, *, explicit=False):
        if False:
            i = 10
            return i + 15
        "Returns the maximum of the matrix or maximum along an axis.\n\n        Args:\n            axis (int): {-2, -1, 0, 1, ``None``} (optional)\n                Axis along which the sum is computed. The default is to\n                compute the maximum over all the matrix elements, returning\n                a scalar (i.e. ``axis`` = ``None``).\n            out (None): (optional)\n                This argument is in the signature *solely* for NumPy\n                compatibility reasons. Do not pass in anything except\n                for the default value, as this argument is not used.\n            explicit (bool): Return the maximum value explicitly specified and\n                ignore all implicit zero entries. If the dimension has no\n                explicit values, a zero is then returned to indicate that it is\n                the only implicit value. This parameter is experimental and may\n                change in the future.\n\n        Returns:\n            (cupy.ndarray or float): Maximum of ``a``. If ``axis`` is\n            ``None``, the result is a scalar value. If ``axis`` is given,\n            the result is an array of dimension ``a.ndim - 1``. This\n            differs from numpy for computational efficiency.\n\n        .. seealso:: min : The minimum value of a sparse matrix along a given\n          axis.\n        .. seealso:: numpy.matrix.max : NumPy's implementation of ``max`` for\n          matrices\n\n        "
        if explicit:
            api_name = 'explicit of cupyx.scipy.sparse.{}.max'.format(self.__class__.__name__)
            _util.experimental(api_name)
        return self._min_or_max(axis, out, cupy.max, explicit)

    def min(self, axis=None, out=None, *, explicit=False):
        if False:
            return 10
        "Returns the minimum of the matrix or maximum along an axis.\n\n        Args:\n            axis (int): {-2, -1, 0, 1, ``None``} (optional)\n                Axis along which the sum is computed. The default is to\n                compute the minimum over all the matrix elements, returning\n                a scalar (i.e. ``axis`` = ``None``).\n            out (None): (optional)\n                This argument is in the signature *solely* for NumPy\n                compatibility reasons. Do not pass in anything except for\n                the default value, as this argument is not used.\n            explicit (bool): Return the minimum value explicitly specified and\n                ignore all implicit zero entries. If the dimension has no\n                explicit values, a zero is then returned to indicate that it is\n                the only implicit value. This parameter is experimental and may\n                change in the future.\n\n        Returns:\n            (cupy.ndarray or float): Minimum of ``a``. If ``axis`` is\n            None, the result is a scalar value. If ``axis`` is given, the\n            result is an array of dimension ``a.ndim - 1``. This differs\n            from numpy for computational efficiency.\n\n        .. seealso:: max : The maximum value of a sparse matrix along a given\n          axis.\n        .. seealso:: numpy.matrix.min : NumPy's implementation of 'min' for\n          matrices\n\n        "
        if explicit:
            api_name = 'explicit of cupyx.scipy.sparse.{}.min'.format(self.__class__.__name__)
            _util.experimental(api_name)
        return self._min_or_max(axis, out, cupy.min, explicit)

    def argmax(self, axis=None, out=None):
        if False:
            while True:
                i = 10
        'Returns indices of maximum elements along an axis.\n\n        Implicit zero elements are taken into account. If there are several\n        maximum values, the index of the first occurrence is returned. If\n        ``NaN`` values occur in the matrix, the output defaults to a zero entry\n        for the row/column in which the NaN occurs.\n\n        Args:\n            axis (int): {-2, -1, 0, 1, ``None``} (optional)\n                Axis along which the argmax is computed. If ``None`` (default),\n                index of the maximum element in the flatten data is returned.\n            out (None): (optional)\n                This argument is in the signature *solely* for NumPy\n                compatibility reasons. Do not pass in anything except for\n                the default value, as this argument is not used.\n\n        Returns:\n            (cupy.narray or int): Indices of maximum elements. If array,\n            its size along ``axis`` is 1.\n\n        '
        return self._arg_min_or_max(axis, out, cupy.argmax, cupy.greater)

    def argmin(self, axis=None, out=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns indices of minimum elements along an axis.\n\n        Implicit zero elements are taken into account. If there are several\n        minimum values, the index of the first occurrence is returned. If\n        ``NaN`` values occur in the matrix, the output defaults to a zero entry\n        for the row/column in which the NaN occurs.\n\n        Args:\n            axis (int): {-2, -1, 0, 1, ``None``} (optional)\n                Axis along which the argmin is computed. If ``None`` (default),\n                index of the minimum element in the flatten data is returned.\n            out (None): (optional)\n                This argument is in the signature *solely* for NumPy\n                compatibility reasons. Do not pass in anything except for\n                the default value, as this argument is not used.\n\n        Returns:\n            (cupy.narray or int): Indices of minimum elements. If matrix,\n            its size along ``axis`` is 1.\n\n        '
        return self._arg_min_or_max(axis, out, cupy.argmin, cupy.less)

def _install_ufunc(func_name):
    if False:
        while True:
            i = 10

    def f(self):
        if False:
            print('Hello World!')
        ufunc = getattr(cupy, func_name)
        result = ufunc(self.data)
        return self._with_data(result)
    f.__doc__ = 'Elementwise %s.' % func_name
    f.__name__ = func_name
    setattr(_data_matrix, func_name, f)

def _install_ufuncs():
    if False:
        return 10
    for func_name in _ufuncs:
        _install_ufunc(func_name)
_install_ufuncs()