from sympy.core.basic import Basic
from sympy.core.containers import Dict, Tuple
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.printing.defaults import Printable
import itertools
from collections.abc import Iterable

class ArrayKind(Kind):
    """
    Kind for N-dimensional array in SymPy.

    This kind represents the multidimensional array that algebraic
    operations are defined. Basic class for this kind is ``NDimArray``,
    but any expression representing the array can have this.

    Parameters
    ==========

    element_kind : Kind
        Kind of the element. Default is :obj:NumberKind `<sympy.core.kind.NumberKind>`,
        which means that the array contains only numbers.

    Examples
    ========

    Any instance of array class has ``ArrayKind``.

    >>> from sympy import NDimArray
    >>> NDimArray([1,2,3]).kind
    ArrayKind(NumberKind)

    Although expressions representing an array may be not instance of
    array class, it will have ``ArrayKind`` as well.

    >>> from sympy import Integral
    >>> from sympy.tensor.array import NDimArray
    >>> from sympy.abc import x
    >>> intA = Integral(NDimArray([1,2,3]), x)
    >>> isinstance(intA, NDimArray)
    False
    >>> intA.kind
    ArrayKind(NumberKind)

    Use ``isinstance()`` to check for ``ArrayKind` without specifying
    the element kind. Use ``is`` with specifying the element kind.

    >>> from sympy.tensor.array import ArrayKind
    >>> from sympy.core import NumberKind
    >>> boolA = NDimArray([True, False])
    >>> isinstance(boolA.kind, ArrayKind)
    True
    >>> boolA.kind is ArrayKind(NumberKind)
    False

    See Also
    ========

    shape : Function to return the shape of objects with ``MatrixKind``.

    """

    def __new__(cls, element_kind=NumberKind):
        if False:
            i = 10
            return i + 15
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'ArrayKind(%s)' % self.element_kind

    @classmethod
    def _union(cls, kinds) -> 'ArrayKind':
        if False:
            print('Hello World!')
        elem_kinds = {e.kind for e in kinds}
        if len(elem_kinds) == 1:
            (elemkind,) = elem_kinds
        else:
            elemkind = UndefinedKind
        return ArrayKind(elemkind)

class NDimArray(Printable):
    """N-dimensional array.

    Examples
    ========

    Create an N-dim array of zeros:

    >>> from sympy import MutableDenseNDimArray
    >>> a = MutableDenseNDimArray.zeros(2, 3, 4)
    >>> a
    [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

    Create an N-dim array from a list;

    >>> a = MutableDenseNDimArray([[2, 3], [4, 5]])
    >>> a
    [[2, 3], [4, 5]]

    >>> b = MutableDenseNDimArray([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    >>> b
    [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]

    Create an N-dim array from a flat list with dimension shape:

    >>> a = MutableDenseNDimArray([1, 2, 3, 4, 5, 6], (2, 3))
    >>> a
    [[1, 2, 3], [4, 5, 6]]

    Create an N-dim array from a matrix:

    >>> from sympy import Matrix
    >>> a = Matrix([[1,2],[3,4]])
    >>> a
    Matrix([
    [1, 2],
    [3, 4]])
    >>> b = MutableDenseNDimArray(a)
    >>> b
    [[1, 2], [3, 4]]

    Arithmetic operations on N-dim arrays

    >>> a = MutableDenseNDimArray([1, 1, 1, 1], (2, 2))
    >>> b = MutableDenseNDimArray([4, 4, 4, 4], (2, 2))
    >>> c = a + b
    >>> c
    [[5, 5], [5, 5]]
    >>> a - b
    [[-3, -3], [-3, -3]]

    """
    _diff_wrt = True
    is_scalar = False

    def __new__(cls, iterable, shape=None, **kwargs):
        if False:
            return 10
        from sympy.tensor.array import ImmutableDenseNDimArray
        return ImmutableDenseNDimArray(iterable, shape, **kwargs)

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        raise NotImplementedError('A subclass of NDimArray should implement __getitem__')

    def _parse_index(self, index):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(index, (SYMPY_INTS, Integer)):
            if index >= self._loop_size:
                raise ValueError('Only a tuple index is accepted')
            return index
        if self._loop_size == 0:
            raise ValueError('Index not valid with an empty array')
        if len(index) != self._rank:
            raise ValueError('Wrong number of array axes')
        real_index = 0
        for i in range(self._rank):
            if index[i] >= self.shape[i] or index[i] < -self.shape[i]:
                raise ValueError('Index ' + str(index) + ' out of border')
            if index[i] < 0:
                real_index += 1
            real_index = real_index * self.shape[i] + index[i]
        return real_index

    def _get_tuple_index(self, integer_index):
        if False:
            for i in range(10):
                print('nop')
        index = []
        for (i, sh) in enumerate(reversed(self.shape)):
            index.append(integer_index % sh)
            integer_index //= sh
        index.reverse()
        return tuple(index)

    def _check_symbolic_index(self, index):
        if False:
            i = 10
            return i + 15
        tuple_index = index if isinstance(index, tuple) else (index,)
        if any((isinstance(i, Expr) and (not i.is_number) for i in tuple_index)):
            for (i, nth_dim) in zip(tuple_index, self.shape):
                if (i < 0) == True or (i >= nth_dim) == True:
                    raise ValueError('index out of range')
            from sympy.tensor import Indexed
            return Indexed(self, *tuple_index)
        return None

    def _setter_iterable_check(self, value):
        if False:
            for i in range(10):
                print('nop')
        from sympy.matrices.matrices import MatrixBase
        if isinstance(value, (Iterable, MatrixBase, NDimArray)):
            raise NotImplementedError

    @classmethod
    def _scan_iterable_shape(cls, iterable):
        if False:
            return 10

        def f(pointer):
            if False:
                print('Hello World!')
            if not isinstance(pointer, Iterable):
                return ([pointer], ())
            if len(pointer) == 0:
                return ([], (0,))
            result = []
            (elems, shapes) = zip(*[f(i) for i in pointer])
            if len(set(shapes)) != 1:
                raise ValueError('could not determine shape unambiguously')
            for i in elems:
                result.extend(i)
            return (result, (len(shapes),) + shapes[0])
        return f(iterable)

    @classmethod
    def _handle_ndarray_creation_inputs(cls, iterable=None, shape=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        from sympy.matrices.matrices import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        if shape is None:
            if iterable is None:
                shape = ()
                iterable = ()
            elif isinstance(iterable, SparseNDimArray):
                return (iterable._shape, iterable._sparse_array)
            elif isinstance(iterable, NDimArray):
                shape = iterable.shape
            elif isinstance(iterable, Iterable):
                (iterable, shape) = cls._scan_iterable_shape(iterable)
            elif isinstance(iterable, MatrixBase):
                shape = iterable.shape
            else:
                shape = ()
                iterable = (iterable,)
        if isinstance(iterable, (Dict, dict)) and shape is not None:
            new_dict = iterable.copy()
            for (k, v) in new_dict.items():
                if isinstance(k, (tuple, Tuple)):
                    new_key = 0
                    for (i, idx) in enumerate(k):
                        new_key = new_key * shape[i] + idx
                    iterable[new_key] = iterable[k]
                    del iterable[k]
        if isinstance(shape, (SYMPY_INTS, Integer)):
            shape = (shape,)
        if not all((isinstance(dim, (SYMPY_INTS, Integer)) for dim in shape)):
            raise TypeError('Shape should contain integers only.')
        return (tuple(shape), iterable)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Overload common function len(). Returns number of elements in array.\n\n        Examples\n        ========\n\n        >>> from sympy import MutableDenseNDimArray\n        >>> a = MutableDenseNDimArray.zeros(3, 3)\n        >>> a\n        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]\n        >>> len(a)\n        9\n\n        '
        return self._loop_size

    @property
    def shape(self):
        if False:
            print('Hello World!')
        '\n        Returns array shape (dimension).\n\n        Examples\n        ========\n\n        >>> from sympy import MutableDenseNDimArray\n        >>> a = MutableDenseNDimArray.zeros(3, 3)\n        >>> a.shape\n        (3, 3)\n\n        '
        return self._shape

    def rank(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns rank of array.\n\n        Examples\n        ========\n\n        >>> from sympy import MutableDenseNDimArray\n        >>> a = MutableDenseNDimArray.zeros(3,4,5,6,3)\n        >>> a.rank()\n        5\n\n        '
        return self._rank

    def diff(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Calculate the derivative of each element in the array.\n\n        Examples\n        ========\n\n        >>> from sympy import ImmutableDenseNDimArray\n        >>> from sympy.abc import x, y\n        >>> M = ImmutableDenseNDimArray([[x, y], [1, x*y]])\n        >>> M.diff(x)\n        [[1, 0], [0, y]]\n\n        '
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        kwargs.setdefault('evaluate', True)
        return ArrayDerivative(self.as_immutable(), *args, **kwargs)

    def _eval_derivative(self, base):
        if False:
            i = 10
            return i + 15
        return self.applyfunc(lambda x: base.diff(x))

    def _eval_derivative_n_times(self, s, n):
        if False:
            while True:
                i = 10
        return Basic._eval_derivative_n_times(self, s, n)

    def applyfunc(self, f):
        if False:
            return 10
        'Apply a function to each element of the N-dim array.\n\n        Examples\n        ========\n\n        >>> from sympy import ImmutableDenseNDimArray\n        >>> m = ImmutableDenseNDimArray([i*2+j for i in range(2) for j in range(2)], (2, 2))\n        >>> m\n        [[0, 1], [2, 3]]\n        >>> m.applyfunc(lambda i: 2*i)\n        [[0, 2], [4, 6]]\n        '
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten
        if isinstance(self, SparseNDimArray) and f(S.Zero) == 0:
            return type(self)({k: f(v) for (k, v) in self._sparse_array.items() if f(v) != 0}, self.shape)
        return type(self)(map(f, Flatten(self)), self.shape)

    def _sympystr(self, printer):
        if False:
            print('Hello World!')

        def f(sh, shape_left, i, j):
            if False:
                i = 10
                return i + 15
            if len(shape_left) == 1:
                return '[' + ', '.join([printer._print(self[self._get_tuple_index(e)]) for e in range(i, j)]) + ']'
            sh //= shape_left[0]
            return '[' + ', '.join([f(sh, shape_left[1:], i + e * sh, i + (e + 1) * sh) for e in range(shape_left[0])]) + ']'
        if self.rank() == 0:
            return printer._print(self[()])
        return f(self._loop_size, self.shape, 0, self._loop_size)

    def tolist(self):
        if False:
            print('Hello World!')
        '\n        Converting MutableDenseNDimArray to one-dim list\n\n        Examples\n        ========\n\n        >>> from sympy import MutableDenseNDimArray\n        >>> a = MutableDenseNDimArray([1, 2, 3, 4], (2, 2))\n        >>> a\n        [[1, 2], [3, 4]]\n        >>> b = a.tolist()\n        >>> b\n        [[1, 2], [3, 4]]\n        '

        def f(sh, shape_left, i, j):
            if False:
                i = 10
                return i + 15
            if len(shape_left) == 1:
                return [self[self._get_tuple_index(e)] for e in range(i, j)]
            result = []
            sh //= shape_left[0]
            for e in range(shape_left[0]):
                result.append(f(sh, shape_left[1:], i + e * sh, i + (e + 1) * sh))
            return result
        return f(self._loop_size, self.shape, 0, self._loop_size)

    def __add__(self, other):
        if False:
            while True:
                i = 10
        from sympy.tensor.array.arrayop import Flatten
        if not isinstance(other, NDimArray):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError('array shape mismatch')
        result_list = [i + j for (i, j) in zip(Flatten(self), Flatten(other))]
        return type(self)(result_list, self.shape)

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        from sympy.tensor.array.arrayop import Flatten
        if not isinstance(other, NDimArray):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError('array shape mismatch')
        result_list = [i - j for (i, j) in zip(Flatten(self), Flatten(other))]
        return type(self)(result_list, self.shape)

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        from sympy.matrices.matrices import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten
        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError('scalar expected, use tensorproduct(...) for tensorial product')
        other = sympify(other)
        if isinstance(self, SparseNDimArray):
            if other.is_zero:
                return type(self)({}, self.shape)
            return type(self)({k: other * v for (k, v) in self._sparse_array.items()}, self.shape)
        result_list = [i * other for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rmul__(self, other):
        if False:
            while True:
                i = 10
        from sympy.matrices.matrices import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten
        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError('scalar expected, use tensorproduct(...) for tensorial product')
        other = sympify(other)
        if isinstance(self, SparseNDimArray):
            if other.is_zero:
                return type(self)({}, self.shape)
            return type(self)({k: other * v for (k, v) in self._sparse_array.items()}, self.shape)
        result_list = [other * i for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __truediv__(self, other):
        if False:
            return 10
        from sympy.matrices.matrices import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten
        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError('scalar expected')
        other = sympify(other)
        if isinstance(self, SparseNDimArray) and other != S.Zero:
            return type(self)({k: v / other for (k, v) in self._sparse_array.items()}, self.shape)
        result_list = [i / other for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rtruediv__(self, other):
        if False:
            return 10
        raise NotImplementedError('unsupported operation on NDimArray')

    def __neg__(self):
        if False:
            print('Hello World!')
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten
        if isinstance(self, SparseNDimArray):
            return type(self)({k: -v for (k, v) in self._sparse_array.items()}, self.shape)
        result_list = [-i for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __iter__(self):
        if False:
            while True:
                i = 10

        def iterator():
            if False:
                while True:
                    i = 10
            if self._shape:
                for i in range(self._shape[0]):
                    yield self[i]
            else:
                yield self[()]
        return iterator()

    def __eq__(self, other):
        if False:
            print('Hello World!')
        '\n        NDimArray instances can be compared to each other.\n        Instances equal if they have same shape and data.\n\n        Examples\n        ========\n\n        >>> from sympy import MutableDenseNDimArray\n        >>> a = MutableDenseNDimArray.zeros(2, 3)\n        >>> b = MutableDenseNDimArray.zeros(2, 3)\n        >>> a == b\n        True\n        >>> c = a.reshape(3, 2)\n        >>> c == b\n        False\n        >>> a[0,0] = 1\n        >>> b[0,0] = 2\n        >>> a == b\n        False\n        '
        from sympy.tensor.array import SparseNDimArray
        if not isinstance(other, NDimArray):
            return False
        if not self.shape == other.shape:
            return False
        if isinstance(self, SparseNDimArray) and isinstance(other, SparseNDimArray):
            return dict(self._sparse_array) == dict(other._sparse_array)
        return list(self) == list(other)

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self == other

    def _eval_transpose(self):
        if False:
            while True:
                i = 10
        if self.rank() != 2:
            raise ValueError('array rank not 2')
        from .arrayop import permutedims
        return permutedims(self, (1, 0))

    def transpose(self):
        if False:
            for i in range(10):
                print('nop')
        return self._eval_transpose()

    def _eval_conjugate(self):
        if False:
            i = 10
            return i + 15
        from sympy.tensor.array.arrayop import Flatten
        return self.func([i.conjugate() for i in Flatten(self)], self.shape)

    def conjugate(self):
        if False:
            for i in range(10):
                print('nop')
        return self._eval_conjugate()

    def _eval_adjoint(self):
        if False:
            for i in range(10):
                print('nop')
        return self.transpose().conjugate()

    def adjoint(self):
        if False:
            i = 10
            return i + 15
        return self._eval_adjoint()

    def _slice_expand(self, s, dim):
        if False:
            print('Hello World!')
        if not isinstance(s, slice):
            return (s,)
        (start, stop, step) = s.indices(dim)
        return [start + i * step for i in range((stop - start) // step)]

    def _get_slice_data_for_array_access(self, index):
        if False:
            i = 10
            return i + 15
        sl_factors = [self._slice_expand(i, dim) for (i, dim) in zip(index, self.shape)]
        eindices = itertools.product(*sl_factors)
        return (sl_factors, eindices)

    def _get_slice_data_for_array_assignment(self, index, value):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, NDimArray):
            value = type(self)(value)
        (sl_factors, eindices) = self._get_slice_data_for_array_access(index)
        slice_offsets = [min(i) if isinstance(i, list) else None for i in sl_factors]
        return (value, eindices, slice_offsets)

    @classmethod
    def _check_special_bounds(cls, flat_list, shape):
        if False:
            for i in range(10):
                print('nop')
        if shape == () and len(flat_list) != 1:
            raise ValueError('arrays without shape need one scalar value')
        if shape == (0,) and len(flat_list) > 0:
            raise ValueError('if array shape is (0,) there cannot be elements')

    def _check_index_for_getitem(self, index):
        if False:
            return 10
        if isinstance(index, (SYMPY_INTS, Integer, slice)):
            index = (index,)
        if len(index) < self.rank():
            index = tuple(index) + tuple((slice(None) for i in range(len(index), self.rank())))
        if len(index) > self.rank():
            raise ValueError('Dimension of index greater than rank of array')
        return index

class ImmutableNDimArray(NDimArray, Basic):
    _op_priority = 11.0

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return Basic.__hash__(self)

    def as_immutable(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def as_mutable(self):
        if False:
            return 10
        raise NotImplementedError('abstract method')