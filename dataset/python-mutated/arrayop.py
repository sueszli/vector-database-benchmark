import itertools
from collections.abc import Iterable
from sympy.core._print_helpers import Printable
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.ndim_array import NDimArray
from sympy.tensor.array.dense_ndim_array import DenseNDimArray, ImmutableDenseNDimArray
from sympy.tensor.array.sparse_ndim_array import SparseNDimArray

def _arrayfy(a):
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices import MatrixBase
    if isinstance(a, NDimArray):
        return a
    if isinstance(a, (MatrixBase, list, tuple, Tuple)):
        return ImmutableDenseNDimArray(a)
    return a

def tensorproduct(*args):
    if False:
        while True:
            i = 10
    '\n    Tensor product among scalars or array-like objects.\n\n    The equivalent operator for array expressions is ``ArrayTensorProduct``,\n    which can be used to keep the expression unevaluated.\n\n    Examples\n    ========\n\n    >>> from sympy.tensor.array import tensorproduct, Array\n    >>> from sympy.abc import x, y, z, t\n    >>> A = Array([[1, 2], [3, 4]])\n    >>> B = Array([x, y])\n    >>> tensorproduct(A, B)\n    [[[x, y], [2*x, 2*y]], [[3*x, 3*y], [4*x, 4*y]]]\n    >>> tensorproduct(A, x)\n    [[x, 2*x], [3*x, 4*x]]\n    >>> tensorproduct(A, B, B)\n    [[[[x**2, x*y], [x*y, y**2]], [[2*x**2, 2*x*y], [2*x*y, 2*y**2]]], [[[3*x**2, 3*x*y], [3*x*y, 3*y**2]], [[4*x**2, 4*x*y], [4*x*y, 4*y**2]]]]\n\n    Applying this function on two matrices will result in a rank 4 array.\n\n    >>> from sympy import Matrix, eye\n    >>> m = Matrix([[x, y], [z, t]])\n    >>> p = tensorproduct(eye(3), m)\n    >>> p\n    [[[[x, y], [z, t]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[x, y], [z, t]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[x, y], [z, t]]]]\n\n    See Also\n    ========\n\n    sympy.tensor.array.expressions.array_expressions.ArrayTensorProduct\n\n    '
    from sympy.tensor.array import SparseNDimArray, ImmutableSparseNDimArray
    if len(args) == 0:
        return S.One
    if len(args) == 1:
        return _arrayfy(args[0])
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    if any((isinstance(arg, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)) for arg in args)):
        return ArrayTensorProduct(*args)
    if len(args) > 2:
        return tensorproduct(tensorproduct(args[0], args[1]), *args[2:])
    (a, b) = map(_arrayfy, args)
    if not isinstance(a, NDimArray) or not isinstance(b, NDimArray):
        return a * b
    if isinstance(a, SparseNDimArray) and isinstance(b, SparseNDimArray):
        lp = len(b)
        new_array = {k1 * lp + k2: v1 * v2 for (k1, v1) in a._sparse_array.items() for (k2, v2) in b._sparse_array.items()}
        return ImmutableSparseNDimArray(new_array, a.shape + b.shape)
    product_list = [i * j for i in Flatten(a) for j in Flatten(b)]
    return ImmutableDenseNDimArray(product_list, a.shape + b.shape)

def _util_contraction_diagonal(array, *contraction_or_diagonal_axes):
    if False:
        while True:
            i = 10
    array = _arrayfy(array)
    taken_dims = set()
    for axes_group in contraction_or_diagonal_axes:
        if not isinstance(axes_group, Iterable):
            raise ValueError('collections of contraction/diagonal axes expected')
        dim = array.shape[axes_group[0]]
        for d in axes_group:
            if d in taken_dims:
                raise ValueError('dimension specified more than once')
            if dim != array.shape[d]:
                raise ValueError('cannot contract or diagonalize between axes of different dimension')
            taken_dims.add(d)
    rank = array.rank()
    remaining_shape = [dim for (i, dim) in enumerate(array.shape) if i not in taken_dims]
    cum_shape = [0] * rank
    _cumul = 1
    for i in range(rank):
        cum_shape[rank - i - 1] = _cumul
        _cumul *= int(array.shape[rank - i - 1])
    remaining_indices = [[cum_shape[i] * j for j in range(array.shape[i])] for i in range(rank) if i not in taken_dims]
    summed_deltas = []
    for axes_group in contraction_or_diagonal_axes:
        lidx = []
        for js in range(array.shape[axes_group[0]]):
            lidx.append(sum([cum_shape[ig] * js for ig in axes_group]))
        summed_deltas.append(lidx)
    return (array, remaining_indices, remaining_shape, summed_deltas)

def tensorcontraction(array, *contraction_axes):
    if False:
        while True:
            i = 10
    '\n    Contraction of an array-like object on the specified axes.\n\n    The equivalent operator for array expressions is ``ArrayContraction``,\n    which can be used to keep the expression unevaluated.\n\n    Examples\n    ========\n\n    >>> from sympy import Array, tensorcontraction\n    >>> from sympy import Matrix, eye\n    >>> tensorcontraction(eye(3), (0, 1))\n    3\n    >>> A = Array(range(18), (3, 2, 3))\n    >>> A\n    [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]]\n    >>> tensorcontraction(A, (0, 2))\n    [21, 30]\n\n    Matrix multiplication may be emulated with a proper combination of\n    ``tensorcontraction`` and ``tensorproduct``\n\n    >>> from sympy import tensorproduct\n    >>> from sympy.abc import a,b,c,d,e,f,g,h\n    >>> m1 = Matrix([[a, b], [c, d]])\n    >>> m2 = Matrix([[e, f], [g, h]])\n    >>> p = tensorproduct(m1, m2)\n    >>> p\n    [[[[a*e, a*f], [a*g, a*h]], [[b*e, b*f], [b*g, b*h]]], [[[c*e, c*f], [c*g, c*h]], [[d*e, d*f], [d*g, d*h]]]]\n    >>> tensorcontraction(p, (1, 2))\n    [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]\n    >>> m1*m2\n    Matrix([\n    [a*e + b*g, a*f + b*h],\n    [c*e + d*g, c*f + d*h]])\n\n    See Also\n    ========\n\n    sympy.tensor.array.expressions.array_expressions.ArrayContraction\n\n    '
    from sympy.tensor.array.expressions.array_expressions import _array_contraction
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    if isinstance(array, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _array_contraction(array, *contraction_axes)
    (array, remaining_indices, remaining_shape, summed_deltas) = _util_contraction_diagonal(array, *contraction_axes)
    contracted_array = []
    for icontrib in itertools.product(*remaining_indices):
        index_base_position = sum(icontrib)
        isum = S.Zero
        for sum_to_index in itertools.product(*summed_deltas):
            idx = array._get_tuple_index(index_base_position + sum(sum_to_index))
            isum += array[idx]
        contracted_array.append(isum)
    if len(remaining_indices) == 0:
        assert len(contracted_array) == 1
        return contracted_array[0]
    return type(array)(contracted_array, remaining_shape)

def tensordiagonal(array, *diagonal_axes):
    if False:
        for i in range(10):
            print('nop')
    '\n    Diagonalization of an array-like object on the specified axes.\n\n    This is equivalent to multiplying the expression by Kronecker deltas\n    uniting the axes.\n\n    The diagonal indices are put at the end of the axes.\n\n    The equivalent operator for array expressions is ``ArrayDiagonal``, which\n    can be used to keep the expression unevaluated.\n\n    Examples\n    ========\n\n    ``tensordiagonal`` acting on a 2-dimensional array by axes 0 and 1 is\n    equivalent to the diagonal of the matrix:\n\n    >>> from sympy import Array, tensordiagonal\n    >>> from sympy import Matrix, eye\n    >>> tensordiagonal(eye(3), (0, 1))\n    [1, 1, 1]\n\n    >>> from sympy.abc import a,b,c,d\n    >>> m1 = Matrix([[a, b], [c, d]])\n    >>> tensordiagonal(m1, [0, 1])\n    [a, d]\n\n    In case of higher dimensional arrays, the diagonalized out dimensions\n    are appended removed and appended as a single dimension at the end:\n\n    >>> A = Array(range(18), (3, 2, 3))\n    >>> A\n    [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]]\n    >>> tensordiagonal(A, (0, 2))\n    [[0, 7, 14], [3, 10, 17]]\n    >>> from sympy import permutedims\n    >>> tensordiagonal(A, (0, 2)) == permutedims(Array([A[0, :, 0], A[1, :, 1], A[2, :, 2]]), [1, 0])\n    True\n\n    See Also\n    ========\n\n    sympy.tensor.array.expressions.array_expressions.ArrayDiagonal\n\n    '
    if any((len(i) <= 1 for i in diagonal_axes)):
        raise ValueError('need at least two axes to diagonalize')
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal, _array_diagonal
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    if isinstance(array, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _array_diagonal(array, *diagonal_axes)
    ArrayDiagonal._validate(array, *diagonal_axes)
    (array, remaining_indices, remaining_shape, diagonal_deltas) = _util_contraction_diagonal(array, *diagonal_axes)
    diagonalized_array = []
    diagonal_shape = [len(i) for i in diagonal_deltas]
    for icontrib in itertools.product(*remaining_indices):
        index_base_position = sum(icontrib)
        isum = []
        for sum_to_index in itertools.product(*diagonal_deltas):
            idx = array._get_tuple_index(index_base_position + sum(sum_to_index))
            isum.append(array[idx])
        isum = type(array)(isum).reshape(*diagonal_shape)
        diagonalized_array.append(isum)
    return type(array)(diagonalized_array, remaining_shape + diagonal_shape)

def derive_by_array(expr, dx):
    if False:
        while True:
            i = 10
    '\n    Derivative by arrays. Supports both arrays and scalars.\n\n    The equivalent operator for array expressions is ``array_derive``.\n\n    Explanation\n    ===========\n\n    Given the array `A_{i_1, \\ldots, i_N}` and the array `X_{j_1, \\ldots, j_M}`\n    this function will return a new array `B` defined by\n\n    `B_{j_1,\\ldots,j_M,i_1,\\ldots,i_N} := \\frac{\\partial A_{i_1,\\ldots,i_N}}{\\partial X_{j_1,\\ldots,j_M}}`\n\n    Examples\n    ========\n\n    >>> from sympy import derive_by_array\n    >>> from sympy.abc import x, y, z, t\n    >>> from sympy import cos\n    >>> derive_by_array(cos(x*t), x)\n    -t*sin(t*x)\n    >>> derive_by_array(cos(x*t), [x, y, z, t])\n    [-t*sin(t*x), 0, 0, -x*sin(t*x)]\n    >>> derive_by_array([x, y**2*z], [[x, y], [z, t]])\n    [[[1, 0], [0, 2*y*z]], [[0, y**2], [0, 0]]]\n\n    '
    from sympy.matrices import MatrixBase
    from sympy.tensor.array import SparseNDimArray
    array_types = (Iterable, MatrixBase, NDimArray)
    if isinstance(dx, array_types):
        dx = ImmutableDenseNDimArray(dx)
        for i in dx:
            if not i._diff_wrt:
                raise ValueError('cannot derive by this array')
    if isinstance(expr, array_types):
        if isinstance(expr, NDimArray):
            expr = expr.as_immutable()
        else:
            expr = ImmutableDenseNDimArray(expr)
        if isinstance(dx, array_types):
            if isinstance(expr, SparseNDimArray):
                lp = len(expr)
                new_array = {k + i * lp: v for (i, x) in enumerate(Flatten(dx)) for (k, v) in expr.diff(x)._sparse_array.items()}
            else:
                new_array = [[y.diff(x) for y in Flatten(expr)] for x in Flatten(dx)]
            return type(expr)(new_array, dx.shape + expr.shape)
        else:
            return expr.diff(dx)
    else:
        expr = _sympify(expr)
        if isinstance(dx, array_types):
            return ImmutableDenseNDimArray([expr.diff(i) for i in Flatten(dx)], dx.shape)
        else:
            dx = _sympify(dx)
            return diff(expr, dx)

def permutedims(expr, perm=None, index_order_old=None, index_order_new=None):
    if False:
        print('Hello World!')
    '\n    Permutes the indices of an array.\n\n    Parameter specifies the permutation of the indices.\n\n    The equivalent operator for array expressions is ``PermuteDims``, which can\n    be used to keep the expression unevaluated.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y, z, t\n    >>> from sympy import sin\n    >>> from sympy import Array, permutedims\n    >>> a = Array([[x, y, z], [t, sin(x), 0]])\n    >>> a\n    [[x, y, z], [t, sin(x), 0]]\n    >>> permutedims(a, (1, 0))\n    [[x, t], [y, sin(x)], [z, 0]]\n\n    If the array is of second order, ``transpose`` can be used:\n\n    >>> from sympy import transpose\n    >>> transpose(a)\n    [[x, t], [y, sin(x)], [z, 0]]\n\n    Examples on higher dimensions:\n\n    >>> b = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])\n    >>> permutedims(b, (2, 1, 0))\n    [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]\n    >>> permutedims(b, (1, 2, 0))\n    [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]\n\n    An alternative way to specify the same permutations as in the previous\n    lines involves passing the *old* and *new* indices, either as a list or as\n    a string:\n\n    >>> permutedims(b, index_order_old="cba", index_order_new="abc")\n    [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]\n    >>> permutedims(b, index_order_old="cab", index_order_new="abc")\n    [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]\n\n    ``Permutation`` objects are also allowed:\n\n    >>> from sympy.combinatorics import Permutation\n    >>> permutedims(b, Permutation([1, 2, 0]))\n    [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]\n\n    See Also\n    ========\n\n    sympy.tensor.array.expressions.array_expressions.PermuteDims\n\n    '
    from sympy.tensor.array import SparseNDimArray
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import _permute_dims
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    from sympy.tensor.array.expressions import PermuteDims
    from sympy.tensor.array.expressions.array_expressions import get_rank
    perm = PermuteDims._get_permutation_from_arguments(perm, index_order_old, index_order_new, get_rank(expr))
    if isinstance(expr, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _permute_dims(expr, perm)
    if not isinstance(expr, NDimArray):
        expr = ImmutableDenseNDimArray(expr)
    from sympy.combinatorics import Permutation
    if not isinstance(perm, Permutation):
        perm = Permutation(list(perm))
    if perm.size != expr.rank():
        raise ValueError('wrong permutation size')
    iperm = ~perm
    new_shape = perm(expr.shape)
    if isinstance(expr, SparseNDimArray):
        return type(expr)({tuple(perm(expr._get_tuple_index(k))): v for (k, v) in expr._sparse_array.items()}, new_shape)
    indices_span = perm([range(i) for i in expr.shape])
    new_array = [None] * len(expr)
    for (i, idx) in enumerate(itertools.product(*indices_span)):
        t = iperm(idx)
        new_array[i] = expr[t]
    return type(expr)(new_array, new_shape)

class Flatten(Printable):
    """
    Flatten an iterable object to a list in a lazy-evaluation way.

    Notes
    =====

    This class is an iterator with which the memory cost can be economised.
    Optimisation has been considered to ameliorate the performance for some
    specific data types like DenseNDimArray and SparseNDimArray.

    Examples
    ========

    >>> from sympy.tensor.array.arrayop import Flatten
    >>> from sympy.tensor.array import Array
    >>> A = Array(range(6)).reshape(2, 3)
    >>> Flatten(A)
    Flatten([[0, 1, 2], [3, 4, 5]])
    >>> [i for i in Flatten(A)]
    [0, 1, 2, 3, 4, 5]
    """

    def __init__(self, iterable):
        if False:
            return 10
        from sympy.matrices.matrices import MatrixBase
        from sympy.tensor.array import NDimArray
        if not isinstance(iterable, (Iterable, MatrixBase)):
            raise NotImplementedError('Data type not yet supported')
        if isinstance(iterable, list):
            iterable = NDimArray(iterable)
        self._iter = iterable
        self._idx = 0

    def __iter__(self):
        if False:
            return 10
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        from sympy.matrices.matrices import MatrixBase
        if len(self._iter) > self._idx:
            if isinstance(self._iter, DenseNDimArray):
                result = self._iter._array[self._idx]
            elif isinstance(self._iter, SparseNDimArray):
                if self._idx in self._iter._sparse_array:
                    result = self._iter._sparse_array[self._idx]
                else:
                    result = 0
            elif isinstance(self._iter, MatrixBase):
                result = self._iter[self._idx]
            elif hasattr(self._iter, '__next__'):
                result = next(self._iter)
            else:
                result = self._iter[self._idx]
        else:
            raise StopIteration
        self._idx += 1
        return result

    def next(self):
        if False:
            return 10
        return self.__next__()

    def _sympystr(self, printer):
        if False:
            i = 10
            return i + 15
        return type(self).__name__ + '(' + printer._print(self._iter) + ')'