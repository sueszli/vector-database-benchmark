import functools
import sys
import math
import warnings
import numpy as np
from .._utils import set_module
import numpy._core.numeric as _nx
from numpy._core.numeric import ScalarType, array
from numpy._core.numerictypes import issubdtype
import numpy.matrixlib as matrixlib
from numpy._core.multiarray import ravel_multi_index, unravel_index
from numpy._core import overrides, linspace
from numpy.lib.stride_tricks import as_strided
from numpy.lib._function_base_impl import diff
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy')
__all__ = ['ravel_multi_index', 'unravel_index', 'mgrid', 'ogrid', 'r_', 'c_', 's_', 'index_exp', 'ix_', 'ndenumerate', 'ndindex', 'fill_diagonal', 'diag_indices', 'diag_indices_from']

def _ix__dispatcher(*args):
    if False:
        while True:
            i = 10
    return args

@array_function_dispatch(_ix__dispatcher)
def ix_(*args):
    if False:
        return 10
    '\n    Construct an open mesh from multiple sequences.\n\n    This function takes N 1-D sequences and returns N outputs with N\n    dimensions each, such that the shape is 1 in all but one dimension\n    and the dimension with the non-unit shape value cycles through all\n    N dimensions.\n\n    Using `ix_` one can quickly construct index arrays that will index\n    the cross product. ``a[np.ix_([1,3],[2,5])]`` returns the array\n    ``[[a[1,2] a[1,5]], [a[3,2] a[3,5]]]``.\n\n    Parameters\n    ----------\n    args : 1-D sequences\n        Each sequence should be of integer or boolean type.\n        Boolean sequences will be interpreted as boolean masks for the\n        corresponding dimension (equivalent to passing in\n        ``np.nonzero(boolean_sequence)``).\n\n    Returns\n    -------\n    out : tuple of ndarrays\n        N arrays with N dimensions each, with N the number of input\n        sequences. Together these arrays form an open mesh.\n\n    See Also\n    --------\n    ogrid, mgrid, meshgrid\n\n    Examples\n    --------\n    >>> a = np.arange(10).reshape(2, 5)\n    >>> a\n    array([[0, 1, 2, 3, 4],\n           [5, 6, 7, 8, 9]])\n    >>> ixgrid = np.ix_([0, 1], [2, 4])\n    >>> ixgrid\n    (array([[0],\n           [1]]), array([[2, 4]]))\n    >>> ixgrid[0].shape, ixgrid[1].shape\n    ((2, 1), (1, 2))\n    >>> a[ixgrid]\n    array([[2, 4],\n           [7, 9]])\n\n    >>> ixgrid = np.ix_([True, True], [2, 4])\n    >>> a[ixgrid]\n    array([[2, 4],\n           [7, 9]])\n    >>> ixgrid = np.ix_([True, True], [False, False, True, False, True])\n    >>> a[ixgrid]\n    array([[2, 4],\n           [7, 9]])\n\n    '
    out = []
    nd = len(args)
    for (k, new) in enumerate(args):
        if not isinstance(new, _nx.ndarray):
            new = np.asarray(new)
            if new.size == 0:
                new = new.astype(_nx.intp)
        if new.ndim != 1:
            raise ValueError('Cross index must be 1 dimensional')
        if issubdtype(new.dtype, _nx.bool_):
            (new,) = new.nonzero()
        new = new.reshape((1,) * k + (new.size,) + (1,) * (nd - k - 1))
        out.append(new)
    return tuple(out)

class nd_grid:
    """
    Construct a multi-dimensional "meshgrid".

    ``grid = nd_grid()`` creates an instance which will return a mesh-grid
    when indexed.  The dimension and number of the output arrays are equal
    to the number of indexing dimensions.  If the step length is not a
    complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then the
    integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    If instantiated with an argument of ``sparse=True``, the mesh-grid is
    open (or not fleshed out) so that only one-dimension of each returned
    argument is greater than 1.

    Parameters
    ----------
    sparse : bool, optional
        Whether the grid is sparse or not. Default is False.

    Notes
    -----
    Two instances of `nd_grid` are made available in the NumPy namespace,
    `mgrid` and `ogrid`, approximately defined as::

        mgrid = nd_grid(sparse=False)
        ogrid = nd_grid(sparse=True)

    Users should use these pre-defined instances instead of using `nd_grid`
    directly.
    """

    def __init__(self, sparse=False):
        if False:
            return 10
        self.sparse = sparse

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        try:
            size = []
            num_list = [0]
            for k in range(len(key)):
                step = key[k].step
                start = key[k].start
                stop = key[k].stop
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, (_nx.complexfloating, complex)):
                    step = abs(step)
                    size.append(int(step))
                else:
                    size.append(int(math.ceil((stop - start) / (step * 1.0))))
                num_list += [start, stop, step]
            typ = _nx.result_type(*num_list)
            if self.sparse:
                nn = [_nx.arange(_x, dtype=_t) for (_x, _t) in zip(size, (typ,) * len(size))]
            else:
                nn = _nx.indices(size, typ)
            for (k, kk) in enumerate(key):
                step = kk.step
                start = kk.start
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, (_nx.complexfloating, complex)):
                    step = int(abs(step))
                    if step != 1:
                        step = (kk.stop - start) / float(step - 1)
                nn[k] = nn[k] * step + start
            if self.sparse:
                slobj = [_nx.newaxis] * len(size)
                for k in range(len(size)):
                    slobj[k] = slice(None, None)
                    nn[k] = nn[k][tuple(slobj)]
                    slobj[k] = _nx.newaxis
            return nn
        except (IndexError, TypeError):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None:
                start = 0
            if isinstance(step, (_nx.complexfloating, complex)):
                step_float = abs(step)
                step = length = int(step_float)
                if step != 1:
                    step = (key.stop - start) / float(step - 1)
                typ = _nx.result_type(start, stop, step_float)
                return _nx.arange(0, length, 1, dtype=typ) * step + start
            else:
                return _nx.arange(start, stop, step)

class MGridClass(nd_grid):
    """
    An instance which returns a dense multi-dimensional "meshgrid".

    An instance which returns a dense (or fleshed out) mesh-grid
    when indexed, so that each returned argument has the same shape.
    The dimensions and number of the output arrays are equal to the
    number of indexing dimensions.  If the step length is not a complex
    number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then
    the integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    Returns
    -------
    mesh-grid
        `ndarray`\\ s all of the same dimensions

    See Also
    --------
    ogrid : like `mgrid` but returns open (not fleshed out) mesh grids
    meshgrid: return coordinate matrices from coordinate vectors
    r_ : array concatenator
    :ref:`how-to-partition`

    Examples
    --------
    >>> np.mgrid[0:5, 0:5]
    array([[[0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4]],
           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]]])
    >>> np.mgrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])

    """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(sparse=False)
mgrid = MGridClass()

class OGridClass(nd_grid):
    """
    An instance which returns an open multi-dimensional "meshgrid".

    An instance which returns an open (i.e. not fleshed out) mesh-grid
    when indexed, so that only one dimension of each returned array is
    greater than 1.  The dimension and number of the output arrays are
    equal to the number of indexing dimensions.  If the step length is
    not a complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then
    the integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    Returns
    -------
    mesh-grid
        `ndarray`\\ s with only one dimension not equal to 1

    See Also
    --------
    mgrid : like `ogrid` but returns dense (or fleshed out) mesh grids
    meshgrid: return coordinate matrices from coordinate vectors
    r_ : array concatenator
    :ref:`how-to-partition`

    Examples
    --------
    >>> from numpy import ogrid
    >>> ogrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])
    >>> ogrid[0:5,0:5]
    [array([[0],
            [1],
            [2],
            [3],
            [4]]), array([[0, 1, 2, 3, 4]])]

    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__(sparse=True)
ogrid = OGridClass()

class AxisConcatenator:
    """
    Translates slice objects to concatenation along an axis.

    For detailed documentation on usage, see `r_`.
    """
    concatenate = staticmethod(_nx.concatenate)
    makemat = staticmethod(matrixlib.matrix)

    def __init__(self, axis=0, matrix=False, ndmin=1, trans1d=-1):
        if False:
            i = 10
            return i + 15
        self.axis = axis
        self.matrix = matrix
        self.trans1d = trans1d
        self.ndmin = ndmin

    def __getitem__(self, key):
        if False:
            return 10
        if isinstance(key, str):
            frame = sys._getframe().f_back
            mymat = matrixlib.bmat(key, frame.f_globals, frame.f_locals)
            return mymat
        if not isinstance(key, tuple):
            key = (key,)
        trans1d = self.trans1d
        ndmin = self.ndmin
        matrix = self.matrix
        axis = self.axis
        objs = []
        result_type_objs = []
        for (k, item) in enumerate(key):
            scalar = False
            if isinstance(item, slice):
                step = item.step
                start = item.start
                stop = item.stop
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, (_nx.complexfloating, complex)):
                    size = int(abs(step))
                    newobj = linspace(start, stop, num=size)
                else:
                    newobj = _nx.arange(start, stop, step)
                if ndmin > 1:
                    newobj = array(newobj, copy=False, ndmin=ndmin)
                    if trans1d != -1:
                        newobj = newobj.swapaxes(-1, trans1d)
            elif isinstance(item, str):
                if k != 0:
                    raise ValueError('special directives must be the first entry.')
                if item in ('r', 'c'):
                    matrix = True
                    col = item == 'c'
                    continue
                if ',' in item:
                    vec = item.split(',')
                    try:
                        (axis, ndmin) = [int(x) for x in vec[:2]]
                        if len(vec) == 3:
                            trans1d = int(vec[2])
                        continue
                    except Exception as e:
                        raise ValueError('unknown special directive {!r}'.format(item)) from e
                try:
                    axis = int(item)
                    continue
                except (ValueError, TypeError) as e:
                    raise ValueError('unknown special directive') from e
            elif type(item) in ScalarType:
                scalar = True
                newobj = item
            else:
                item_ndim = np.ndim(item)
                newobj = array(item, copy=False, subok=True, ndmin=ndmin)
                if trans1d != -1 and item_ndim < ndmin:
                    k2 = ndmin - item_ndim
                    k1 = trans1d
                    if k1 < 0:
                        k1 += k2 + 1
                    defaxes = list(range(ndmin))
                    axes = defaxes[:k1] + defaxes[k2:] + defaxes[k1:k2]
                    newobj = newobj.transpose(axes)
            objs.append(newobj)
            if scalar:
                result_type_objs.append(item)
            else:
                result_type_objs.append(newobj.dtype)
        if len(result_type_objs) != 0:
            final_dtype = _nx.result_type(*result_type_objs)
            objs = [array(obj, copy=False, subok=True, ndmin=ndmin, dtype=final_dtype) for obj in objs]
        res = self.concatenate(tuple(objs), axis=axis)
        if matrix:
            oldndim = res.ndim
            res = self.makemat(res)
            if oldndim == 1 and col:
                res = res.T
        return res

    def __len__(self):
        if False:
            while True:
                i = 10
        return 0

class RClass(AxisConcatenator):
    """
    Translates slice objects to concatenation along the first axis.

    This is a simple way to build up arrays quickly. There are two use cases.

    1. If the index expression contains comma separated arrays, then stack
       them along their first axis.
    2. If the index expression contains slice notation or scalars then create
       a 1-D array with a range indicated by the slice notation.

    If slice notation is used, the syntax ``start:stop:step`` is equivalent
    to ``np.arange(start, stop, step)`` inside of the brackets. However, if
    ``step`` is an imaginary number (i.e. 100j) then its integer portion is
    interpreted as a number-of-points desired and the start and stop are
    inclusive. In other words ``start:stop:stepj`` is interpreted as
    ``np.linspace(start, stop, step, endpoint=1)`` inside of the brackets.
    After expansion of slice notation, all comma separated sequences are
    concatenated together.

    Optional character strings placed as the first element of the index
    expression can be used to change the output. The strings 'r' or 'c' result
    in matrix output. If the result is 1-D and 'r' is specified a 1 x N (row)
    matrix is produced. If the result is 1-D and 'c' is specified, then a N x 1
    (column) matrix is produced. If the result is 2-D then both provide the
    same matrix result.

    A string integer specifies which axis to stack multiple comma separated
    arrays along. A string of two comma-separated integers allows indication
    of the minimum number of dimensions to force each entry into as the
    second integer (the axis to concatenate along is still the first integer).

    A string with three comma-separated integers allows specification of the
    axis to concatenate along, the minimum number of dimensions to force the
    entries to, and which axis should contain the start of the arrays which
    are less than the specified number of dimensions. In other words the third
    integer allows you to specify where the 1's should be placed in the shape
    of the arrays that have their shapes upgraded. By default, they are placed
    in the front of the shape tuple. The third argument allows you to specify
    where the start of the array should be instead. Thus, a third argument of
    '0' would place the 1's at the end of the array shape. Negative integers
    specify where in the new shape tuple the last dimension of upgraded arrays
    should be placed, so the default is '-1'.

    Parameters
    ----------
    Not a function, so takes no parameters


    Returns
    -------
    A concatenated ndarray or matrix.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    c_ : Translates slice objects to concatenation along the second axis.

    Examples
    --------
    >>> np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])]
    array([1, 2, 3, ..., 4, 5, 6])
    >>> np.r_[-1:1:6j, [0]*3, 5, 6]
    array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ,  0. ,  0. ,  0. ,  5. ,  6. ])

    String integers specify the axis to concatenate along or the minimum
    number of dimensions to force entries into.

    >>> a = np.array([[0, 1, 2], [3, 4, 5]])
    >>> np.r_['-1', a, a] # concatenate along last axis
    array([[0, 1, 2, 0, 1, 2],
           [3, 4, 5, 3, 4, 5]])
    >>> np.r_['0,2', [1,2,3], [4,5,6]] # concatenate along first axis, dim>=2
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> np.r_['0,2,0', [1,2,3], [4,5,6]]
    array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]])
    >>> np.r_['1,2,0', [1,2,3], [4,5,6]]
    array([[1, 4],
           [2, 5],
           [3, 6]])

    Using 'r' or 'c' as a first string argument creates a matrix.

    >>> np.r_['r',[1,2,3], [4,5,6]]
    matrix([[1, 2, 3, 4, 5, 6]])

    """

    def __init__(self):
        if False:
            return 10
        AxisConcatenator.__init__(self, 0)
r_ = RClass()

class CClass(AxisConcatenator):
    """
    Translates slice objects to concatenation along the second axis.

    This is short-hand for ``np.r_['-1,2,0', index expression]``, which is
    useful because of its common occurrence. In particular, arrays will be
    stacked along their last axis after being upgraded to at least 2-D with
    1's post-pended to the shape (column vectors made out of 1-D arrays).

    See Also
    --------
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    r_ : For more detailed documentation.

    Examples
    --------
    >>> np.c_[np.array([1,2,3]), np.array([4,5,6])]
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
    array([[1, 2, 3, ..., 4, 5, 6]])

    """

    def __init__(self):
        if False:
            while True:
                i = 10
        AxisConcatenator.__init__(self, -1, ndmin=2, trans1d=0)
c_ = CClass()

@set_module('numpy')
class ndenumerate:
    """
    Multidimensional index iterator.

    Return an iterator yielding pairs of array coordinates and values.

    Parameters
    ----------
    arr : ndarray
      Input array.

    See Also
    --------
    ndindex, flatiter

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> for index, x in np.ndenumerate(a):
    ...     print(index, x)
    (0, 0) 1
    (0, 1) 2
    (1, 0) 3
    (1, 1) 4

    """

    def __init__(self, arr):
        if False:
            for i in range(10):
                print('nop')
        self.iter = np.asarray(arr).flat

    def __next__(self):
        if False:
            i = 10
            return i + 15
        '\n        Standard iterator method, returns the index tuple and array value.\n\n        Returns\n        -------\n        coords : tuple of ints\n            The indices of the current iteration.\n        val : scalar\n            The array element of the current iteration.\n\n        '
        return (self.iter.coords, next(self.iter))

    def __iter__(self):
        if False:
            return 10
        return self

@set_module('numpy')
class ndindex:
    """
    An N-dimensional iterator object to index arrays.

    Given the shape of an array, an `ndindex` instance iterates over
    the N-dimensional index of the array. At each iteration a tuple
    of indices is returned, the last dimension is iterated over first.

    Parameters
    ----------
    shape : ints, or a single tuple of ints
        The size of each dimension of the array can be passed as
        individual parameters or as the elements of a tuple.

    See Also
    --------
    ndenumerate, flatiter

    Examples
    --------
    Dimensions as individual arguments

    >>> for index in np.ndindex(3, 2, 1):
    ...     print(index)
    (0, 0, 0)
    (0, 1, 0)
    (1, 0, 0)
    (1, 1, 0)
    (2, 0, 0)
    (2, 1, 0)

    Same dimensions - but in a tuple ``(3, 2, 1)``

    >>> for index in np.ndindex((3, 2, 1)):
    ...     print(index)
    (0, 0, 0)
    (0, 1, 0)
    (1, 0, 0)
    (1, 1, 0)
    (2, 0, 0)
    (2, 1, 0)

    """

    def __init__(self, *shape):
        if False:
            while True:
                i = 10
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        x = as_strided(_nx.zeros(1), shape=shape, strides=_nx.zeros_like(shape))
        self._it = _nx.nditer(x, flags=['multi_index', 'zerosize_ok'], order='C')

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

    def ndincr(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Increment the multi-dimensional index by one.\n\n        This method is for backward compatibility only: do not use.\n\n        .. deprecated:: 1.20.0\n            This method has been advised against since numpy 1.8.0, but only\n            started emitting DeprecationWarning as of this version.\n        '
        warnings.warn('`ndindex.ndincr()` is deprecated, use `next(ndindex)` instead', DeprecationWarning, stacklevel=2)
        next(self)

    def __next__(self):
        if False:
            return 10
        '\n        Standard iterator method, updates the index and returns the index\n        tuple.\n\n        Returns\n        -------\n        val : tuple of ints\n            Returns a tuple containing the indices of the current\n            iteration.\n\n        '
        next(self._it)
        return self._it.multi_index

class IndexExpression:
    """
    A nicer way to build up index tuples for arrays.

    .. note::
       Use one of the two predefined instances ``index_exp`` or `s_`
       rather than directly using `IndexExpression`.

    For any index combination, including slicing and axis insertion,
    ``a[indices]`` is the same as ``a[np.index_exp[indices]]`` for any
    array `a`. However, ``np.index_exp[indices]`` can be used anywhere
    in Python code and returns a tuple of slice objects that can be
    used in the construction of complex index expressions.

    Parameters
    ----------
    maketuple : bool
        If True, always returns a tuple.

    See Also
    --------
    s_ : Predefined instance without tuple conversion:
       `s_ = IndexExpression(maketuple=False)`.
       The ``index_exp`` is another predefined instance that
       always returns a tuple:
       `index_exp = IndexExpression(maketuple=True)`.

    Notes
    -----
    You can do all this with :class:`slice` plus a few special objects,
    but there's a lot to remember and this version is simpler because
    it uses the standard array indexing syntax.

    Examples
    --------
    >>> np.s_[2::2]
    slice(2, None, 2)
    >>> np.index_exp[2::2]
    (slice(2, None, 2),)

    >>> np.array([0, 1, 2, 3, 4])[np.s_[2::2]]
    array([2, 4])

    """

    def __init__(self, maketuple):
        if False:
            for i in range(10):
                print('nop')
        self.maketuple = maketuple

    def __getitem__(self, item):
        if False:
            return 10
        if self.maketuple and (not isinstance(item, tuple)):
            return (item,)
        else:
            return item
index_exp = IndexExpression(maketuple=True)
s_ = IndexExpression(maketuple=False)

def _fill_diagonal_dispatcher(a, val, wrap=None):
    if False:
        print('Hello World!')
    return (a,)

@array_function_dispatch(_fill_diagonal_dispatcher)
def fill_diagonal(a, val, wrap=False):
    if False:
        while True:
            i = 10
    'Fill the main diagonal of the given array of any dimensionality.\n\n    For an array `a` with ``a.ndim >= 2``, the diagonal is the list of\n    locations with indices ``a[i, ..., i]`` all identical. This function\n    modifies the input array in-place, it does not return a value.\n\n    Parameters\n    ----------\n    a : array, at least 2-D.\n      Array whose diagonal is to be filled, it gets modified in-place.\n\n    val : scalar or array_like\n      Value(s) to write on the diagonal. If `val` is scalar, the value is\n      written along the diagonal. If array-like, the flattened `val` is\n      written along the diagonal, repeating if necessary to fill all\n      diagonal entries.\n\n    wrap : bool\n      For tall matrices in NumPy version up to 1.6.2, the\n      diagonal "wrapped" after N columns. You can have this behavior\n      with this option. This affects only tall matrices.\n\n    See also\n    --------\n    diag_indices, diag_indices_from\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    This functionality can be obtained via `diag_indices`, but internally\n    this version uses a much faster implementation that never constructs the\n    indices and uses simple slicing.\n\n    Examples\n    --------\n    >>> a = np.zeros((3, 3), int)\n    >>> np.fill_diagonal(a, 5)\n    >>> a\n    array([[5, 0, 0],\n           [0, 5, 0],\n           [0, 0, 5]])\n\n    The same function can operate on a 4-D array:\n\n    >>> a = np.zeros((3, 3, 3, 3), int)\n    >>> np.fill_diagonal(a, 4)\n\n    We only show a few blocks for clarity:\n\n    >>> a[0, 0]\n    array([[4, 0, 0],\n           [0, 0, 0],\n           [0, 0, 0]])\n    >>> a[1, 1]\n    array([[0, 0, 0],\n           [0, 4, 0],\n           [0, 0, 0]])\n    >>> a[2, 2]\n    array([[0, 0, 0],\n           [0, 0, 0],\n           [0, 0, 4]])\n\n    The wrap option affects only tall matrices:\n\n    >>> # tall matrices no wrap\n    >>> a = np.zeros((5, 3), int)\n    >>> np.fill_diagonal(a, 4)\n    >>> a\n    array([[4, 0, 0],\n           [0, 4, 0],\n           [0, 0, 4],\n           [0, 0, 0],\n           [0, 0, 0]])\n\n    >>> # tall matrices wrap\n    >>> a = np.zeros((5, 3), int)\n    >>> np.fill_diagonal(a, 4, wrap=True)\n    >>> a\n    array([[4, 0, 0],\n           [0, 4, 0],\n           [0, 0, 4],\n           [0, 0, 0],\n           [4, 0, 0]])\n\n    >>> # wide matrices\n    >>> a = np.zeros((3, 5), int)\n    >>> np.fill_diagonal(a, 4, wrap=True)\n    >>> a\n    array([[4, 0, 0, 0, 0],\n           [0, 4, 0, 0, 0],\n           [0, 0, 4, 0, 0]])\n\n    The anti-diagonal can be filled by reversing the order of elements\n    using either `numpy.flipud` or `numpy.fliplr`.\n\n    >>> a = np.zeros((3, 3), int);\n    >>> np.fill_diagonal(np.fliplr(a), [1,2,3])  # Horizontal flip\n    >>> a\n    array([[0, 0, 1],\n           [0, 2, 0],\n           [3, 0, 0]])\n    >>> np.fill_diagonal(np.flipud(a), [1,2,3])  # Vertical flip\n    >>> a\n    array([[0, 0, 3],\n           [0, 2, 0],\n           [1, 0, 0]])\n\n    Note that the order in which the diagonal is filled varies depending\n    on the flip function.\n    '
    if a.ndim < 2:
        raise ValueError('array must be at least 2-d')
    end = None
    if a.ndim == 2:
        step = a.shape[1] + 1
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        if not np.all(diff(a.shape) == 0):
            raise ValueError('All dimensions of input must be of equal length')
        step = 1 + np.cumprod(a.shape[:-1]).sum()
    a.flat[:end:step] = val

@set_module('numpy')
def diag_indices(n, ndim=2):
    if False:
        return 10
    '\n    Return the indices to access the main diagonal of an array.\n\n    This returns a tuple of indices that can be used to access the main\n    diagonal of an array `a` with ``a.ndim >= 2`` dimensions and shape\n    (n, n, ..., n). For ``a.ndim = 2`` this is the usual diagonal, for\n    ``a.ndim > 2`` this is the set of indices to access ``a[i, i, ..., i]``\n    for ``i = [0..n-1]``.\n\n    Parameters\n    ----------\n    n : int\n      The size, along each dimension, of the arrays for which the returned\n      indices can be used.\n\n    ndim : int, optional\n      The number of dimensions.\n\n    See Also\n    --------\n    diag_indices_from\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    Examples\n    --------\n    Create a set of indices to access the diagonal of a (4, 4) array:\n\n    >>> di = np.diag_indices(4)\n    >>> di\n    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))\n    >>> a = np.arange(16).reshape(4, 4)\n    >>> a\n    array([[ 0,  1,  2,  3],\n           [ 4,  5,  6,  7],\n           [ 8,  9, 10, 11],\n           [12, 13, 14, 15]])\n    >>> a[di] = 100\n    >>> a\n    array([[100,   1,   2,   3],\n           [  4, 100,   6,   7],\n           [  8,   9, 100,  11],\n           [ 12,  13,  14, 100]])\n\n    Now, we create indices to manipulate a 3-D array:\n\n    >>> d3 = np.diag_indices(2, 3)\n    >>> d3\n    (array([0, 1]), array([0, 1]), array([0, 1]))\n\n    And use it to set the diagonal of an array of zeros to 1:\n\n    >>> a = np.zeros((2, 2, 2), dtype=int)\n    >>> a[d3] = 1\n    >>> a\n    array([[[1, 0],\n            [0, 0]],\n           [[0, 0],\n            [0, 1]]])\n\n    '
    idx = np.arange(n)
    return (idx,) * ndim

def _diag_indices_from(arr):
    if False:
        while True:
            i = 10
    return (arr,)

@array_function_dispatch(_diag_indices_from)
def diag_indices_from(arr):
    if False:
        return 10
    '\n    Return the indices to access the main diagonal of an n-dimensional array.\n\n    See `diag_indices` for full details.\n\n    Parameters\n    ----------\n    arr : array, at least 2-D\n\n    See Also\n    --------\n    diag_indices\n\n    Notes\n    -----\n    .. versionadded:: 1.4.0\n\n    Examples\n    --------\n    \n    Create a 4 by 4 array.\n\n    >>> a = np.arange(16).reshape(4, 4)\n    >>> a\n    array([[ 0,  1,  2,  3],\n           [ 4,  5,  6,  7],\n           [ 8,  9, 10, 11],\n           [12, 13, 14, 15]])\n    \n    Get the indices of the diagonal elements.\n\n    >>> di = np.diag_indices_from(a)\n    >>> di\n    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))\n\n    >>> a[di]\n    array([ 0,  5, 10, 15])\n\n    This is simply syntactic sugar for diag_indices.\n\n    >>> np.diag_indices(a.shape[0])\n    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))\n\n    '
    if not arr.ndim >= 2:
        raise ValueError('input array must be at least 2-d')
    if not np.all(diff(arr.shape) == 0):
        raise ValueError('All dimensions of input must be of equal length')
    return diag_indices(arr.shape[0], arr.ndim)