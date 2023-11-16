from sympy.core.basic import Basic
from sympy.core.containers import Dict, Tuple
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.mutable_ndim_array import MutableNDimArray
from sympy.tensor.array.ndim_array import NDimArray, ImmutableNDimArray
from sympy.utilities.iterables import flatten
import functools

class SparseNDimArray(NDimArray):

    def __new__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return ImmutableSparseNDimArray(*args, **kwargs)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get an element from a sparse N-dim array.\n\n        Examples\n        ========\n\n        >>> from sympy import MutableSparseNDimArray\n        >>> a = MutableSparseNDimArray(range(4), (2, 2))\n        >>> a\n        [[0, 1], [2, 3]]\n        >>> a[0, 0]\n        0\n        >>> a[1, 1]\n        3\n        >>> a[0]\n        [0, 1]\n        >>> a[1]\n        [2, 3]\n\n        Symbolic indexing:\n\n        >>> from sympy.abc import i, j\n        >>> a[i, j]\n        [[0, 1], [2, 3]][i, j]\n\n        Replace `i` and `j` to get element `(0, 0)`:\n\n        >>> a[i, j].subs({i: 0, j: 0})\n        0\n\n        '
        syindex = self._check_symbolic_index(index)
        if syindex is not None:
            return syindex
        index = self._check_index_for_getitem(index)
        if isinstance(index, tuple) and any((isinstance(i, slice) for i in index)):
            (sl_factors, eindices) = self._get_slice_data_for_array_access(index)
            array = [self._sparse_array.get(self._parse_index(i), S.Zero) for i in eindices]
            nshape = [len(el) for (i, el) in enumerate(sl_factors) if isinstance(index[i], slice)]
            return type(self)(array, nshape)
        else:
            index = self._parse_index(index)
            return self._sparse_array.get(index, S.Zero)

    @classmethod
    def zeros(cls, *shape):
        if False:
            i = 10
            return i + 15
        '\n        Return a sparse N-dim array of zeros.\n        '
        return cls({}, shape)

    def tomatrix(self):
        if False:
            i = 10
            return i + 15
        '\n        Converts MutableDenseNDimArray to Matrix. Can convert only 2-dim array, else will raise error.\n\n        Examples\n        ========\n\n        >>> from sympy import MutableSparseNDimArray\n        >>> a = MutableSparseNDimArray([1 for i in range(9)], (3, 3))\n        >>> b = a.tomatrix()\n        >>> b\n        Matrix([\n        [1, 1, 1],\n        [1, 1, 1],\n        [1, 1, 1]])\n        '
        from sympy.matrices import SparseMatrix
        if self.rank() != 2:
            raise ValueError('Dimensions must be of size of 2')
        mat_sparse = {}
        for (key, value) in self._sparse_array.items():
            mat_sparse[self._get_tuple_index(key)] = value
        return SparseMatrix(self.shape[0], self.shape[1], mat_sparse)

    def reshape(self, *newshape):
        if False:
            return 10
        new_total_size = functools.reduce(lambda x, y: x * y, newshape)
        if new_total_size != self._loop_size:
            raise ValueError('Invalid reshape parameters ' + newshape)
        return type(self)(self._sparse_array, newshape)

class ImmutableSparseNDimArray(SparseNDimArray, ImmutableNDimArray):

    def __new__(cls, iterable=None, shape=None, **kwargs):
        if False:
            while True:
                i = 10
        (shape, flat_list) = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        shape = Tuple(*map(_sympify, shape))
        cls._check_special_bounds(flat_list, shape)
        loop_size = functools.reduce(lambda x, y: x * y, shape) if shape else len(flat_list)
        if isinstance(flat_list, (dict, Dict)):
            sparse_array = Dict(flat_list)
        else:
            sparse_array = {}
            for (i, el) in enumerate(flatten(flat_list)):
                if el != 0:
                    sparse_array[i] = _sympify(el)
        sparse_array = Dict(sparse_array)
        self = Basic.__new__(cls, sparse_array, shape, **kwargs)
        self._shape = shape
        self._rank = len(shape)
        self._loop_size = loop_size
        self._sparse_array = sparse_array
        return self

    def __setitem__(self, index, value):
        if False:
            return 10
        raise TypeError('immutable N-dim array')

    def as_mutable(self):
        if False:
            while True:
                i = 10
        return MutableSparseNDimArray(self)

class MutableSparseNDimArray(MutableNDimArray, SparseNDimArray):

    def __new__(cls, iterable=None, shape=None, **kwargs):
        if False:
            print('Hello World!')
        (shape, flat_list) = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        self = object.__new__(cls)
        self._shape = shape
        self._rank = len(shape)
        self._loop_size = functools.reduce(lambda x, y: x * y, shape) if shape else len(flat_list)
        if isinstance(flat_list, (dict, Dict)):
            self._sparse_array = dict(flat_list)
            return self
        self._sparse_array = {}
        for (i, el) in enumerate(flatten(flat_list)):
            if el != 0:
                self._sparse_array[i] = _sympify(el)
        return self

    def __setitem__(self, index, value):
        if False:
            return 10
        'Allows to set items to MutableDenseNDimArray.\n\n        Examples\n        ========\n\n        >>> from sympy import MutableSparseNDimArray\n        >>> a = MutableSparseNDimArray.zeros(2, 2)\n        >>> a[0, 0] = 1\n        >>> a[1, 1] = 1\n        >>> a\n        [[1, 0], [0, 1]]\n        '
        if isinstance(index, tuple) and any((isinstance(i, slice) for i in index)):
            (value, eindices, slice_offsets) = self._get_slice_data_for_array_assignment(index, value)
            for i in eindices:
                other_i = [ind - j for (ind, j) in zip(i, slice_offsets) if j is not None]
                other_value = value[other_i]
                complete_index = self._parse_index(i)
                if other_value != 0:
                    self._sparse_array[complete_index] = other_value
                elif complete_index in self._sparse_array:
                    self._sparse_array.pop(complete_index)
        else:
            index = self._parse_index(index)
            value = _sympify(value)
            if value == 0 and index in self._sparse_array:
                self._sparse_array.pop(index)
            else:
                self._sparse_array[index] = value

    def as_immutable(self):
        if False:
            i = 10
            return i + 15
        return ImmutableSparseNDimArray(self)

    @property
    def free_symbols(self):
        if False:
            for i in range(10):
                print('nop')
        return {i for j in self._sparse_array.values() for i in j.free_symbols}