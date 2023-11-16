import functools
from typing import List
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.mutable_ndim_array import MutableNDimArray
from sympy.tensor.array.ndim_array import NDimArray, ImmutableNDimArray, ArrayKind
from sympy.utilities.iterables import flatten

class DenseNDimArray(NDimArray):
    _array: List[Basic]

    def __new__(self, *args, **kwargs):
        if False:
            return 10
        return ImmutableDenseNDimArray(*args, **kwargs)

    @property
    def kind(self) -> ArrayKind:
        if False:
            i = 10
            return i + 15
        return ArrayKind._union(self._array)

    def __getitem__(self, index):
        if False:
            return 10
        '\n        Allows to get items from N-dim array.\n\n        Examples\n        ========\n\n        >>> from sympy import MutableDenseNDimArray\n        >>> a = MutableDenseNDimArray([0, 1, 2, 3], (2, 2))\n        >>> a\n        [[0, 1], [2, 3]]\n        >>> a[0, 0]\n        0\n        >>> a[1, 1]\n        3\n        >>> a[0]\n        [0, 1]\n        >>> a[1]\n        [2, 3]\n\n\n        Symbolic index:\n\n        >>> from sympy.abc import i, j\n        >>> a[i, j]\n        [[0, 1], [2, 3]][i, j]\n\n        Replace `i` and `j` to get element `(1, 1)`:\n\n        >>> a[i, j].subs({i: 1, j: 1})\n        3\n\n        '
        syindex = self._check_symbolic_index(index)
        if syindex is not None:
            return syindex
        index = self._check_index_for_getitem(index)
        if isinstance(index, tuple) and any((isinstance(i, slice) for i in index)):
            (sl_factors, eindices) = self._get_slice_data_for_array_access(index)
            array = [self._array[self._parse_index(i)] for i in eindices]
            nshape = [len(el) for (i, el) in enumerate(sl_factors) if isinstance(index[i], slice)]
            return type(self)(array, nshape)
        else:
            index = self._parse_index(index)
            return self._array[index]

    @classmethod
    def zeros(cls, *shape):
        if False:
            for i in range(10):
                print('nop')
        list_length = functools.reduce(lambda x, y: x * y, shape, S.One)
        return cls._new(([0] * list_length,), shape)

    def tomatrix(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts MutableDenseNDimArray to Matrix. Can convert only 2-dim array, else will raise error.\n\n        Examples\n        ========\n\n        >>> from sympy import MutableDenseNDimArray\n        >>> a = MutableDenseNDimArray([1 for i in range(9)], (3, 3))\n        >>> b = a.tomatrix()\n        >>> b\n        Matrix([\n        [1, 1, 1],\n        [1, 1, 1],\n        [1, 1, 1]])\n\n        '
        from sympy.matrices import Matrix
        if self.rank() != 2:
            raise ValueError('Dimensions must be of size of 2')
        return Matrix(self.shape[0], self.shape[1], self._array)

    def reshape(self, *newshape):
        if False:
            while True:
                i = 10
        '\n        Returns MutableDenseNDimArray instance with new shape. Elements number\n        must be        suitable to new shape. The only argument of method sets\n        new shape.\n\n        Examples\n        ========\n\n        >>> from sympy import MutableDenseNDimArray\n        >>> a = MutableDenseNDimArray([1, 2, 3, 4, 5, 6], (2, 3))\n        >>> a.shape\n        (2, 3)\n        >>> a\n        [[1, 2, 3], [4, 5, 6]]\n        >>> b = a.reshape(3, 2)\n        >>> b.shape\n        (3, 2)\n        >>> b\n        [[1, 2], [3, 4], [5, 6]]\n\n        '
        new_total_size = functools.reduce(lambda x, y: x * y, newshape)
        if new_total_size != self._loop_size:
            raise ValueError('Expecting reshape size to %d but got prod(%s) = %d' % (self._loop_size, str(newshape), new_total_size))
        return type(self)(self._array, newshape)

class ImmutableDenseNDimArray(DenseNDimArray, ImmutableNDimArray):

    def __new__(cls, iterable, shape=None, **kwargs):
        if False:
            while True:
                i = 10
        return cls._new(iterable, shape, **kwargs)

    @classmethod
    def _new(cls, iterable, shape, **kwargs):
        if False:
            while True:
                i = 10
        (shape, flat_list) = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        shape = Tuple(*map(_sympify, shape))
        cls._check_special_bounds(flat_list, shape)
        flat_list = flatten(flat_list)
        flat_list = Tuple(*flat_list)
        self = Basic.__new__(cls, flat_list, shape, **kwargs)
        self._shape = shape
        self._array = list(flat_list)
        self._rank = len(shape)
        self._loop_size = functools.reduce(lambda x, y: x * y, shape, 1)
        return self

    def __setitem__(self, index, value):
        if False:
            print('Hello World!')
        raise TypeError('immutable N-dim array')

    def as_mutable(self):
        if False:
            return 10
        return MutableDenseNDimArray(self)

    def _eval_simplify(self, **kwargs):
        if False:
            return 10
        from sympy.simplify.simplify import simplify
        return self.applyfunc(simplify)

class MutableDenseNDimArray(DenseNDimArray, MutableNDimArray):

    def __new__(cls, iterable=None, shape=None, **kwargs):
        if False:
            print('Hello World!')
        return cls._new(iterable, shape, **kwargs)

    @classmethod
    def _new(cls, iterable, shape, **kwargs):
        if False:
            return 10
        (shape, flat_list) = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        flat_list = flatten(flat_list)
        self = object.__new__(cls)
        self._shape = shape
        self._array = list(flat_list)
        self._rank = len(shape)
        self._loop_size = functools.reduce(lambda x, y: x * y, shape) if shape else len(flat_list)
        return self

    def __setitem__(self, index, value):
        if False:
            print('Hello World!')
        'Allows to set items to MutableDenseNDimArray.\n\n        Examples\n        ========\n\n        >>> from sympy import MutableDenseNDimArray\n        >>> a = MutableDenseNDimArray.zeros(2,  2)\n        >>> a[0,0] = 1\n        >>> a[1,1] = 1\n        >>> a\n        [[1, 0], [0, 1]]\n\n        '
        if isinstance(index, tuple) and any((isinstance(i, slice) for i in index)):
            (value, eindices, slice_offsets) = self._get_slice_data_for_array_assignment(index, value)
            for i in eindices:
                other_i = [ind - j for (ind, j) in zip(i, slice_offsets) if j is not None]
                self._array[self._parse_index(i)] = value[other_i]
        else:
            index = self._parse_index(index)
            self._setter_iterable_check(value)
            value = _sympify(value)
            self._array[index] = value

    def as_immutable(self):
        if False:
            while True:
                i = 10
        return ImmutableDenseNDimArray(self)

    @property
    def free_symbols(self):
        if False:
            print('Hello World!')
        return {i for j in self._array for i in j.free_symbols}