"""
An ndarray subclass for working with arrays of strings.
"""
from functools import partial, total_ordering
from operator import eq, ne
import re
import numpy as np
from numpy import ndarray
import pandas as pd
from toolz import compose
from zipline.utils.compat import unicode
from zipline.utils.functional import instance
from zipline.utils.preprocess import preprocess
from zipline.utils.sentinel import sentinel
from zipline.utils.input_validation import coerce, expect_kinds, expect_types, optional
from zipline.utils.numpy_utils import bool_dtype, unsigned_int_dtype_with_size_in_bytes, is_object, object_dtype
from zipline.utils.pandas_utils import ignore_pandas_nan_categorical_warning
from ._factorize import factorize_strings, factorize_strings_known_categories, smallest_uint_that_can_hold

def compare_arrays(left, right):
    if False:
        for i in range(10):
            print('nop')
    'Eq check with a short-circuit for identical objects.'
    return left is right or (left.shape == right.shape and (left == right).all())

def _make_unsupported_method(name):
    if False:
        return 10

    def method(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Method %s is not supported on LabelArrays.' % name)
    method.__name__ = name
    method.__doc__ = 'Unsupported LabelArray Method: %s' % name
    return method

class MissingValueMismatch(ValueError):
    """
    Error raised on attempt to perform operations between LabelArrays with
    mismatched missing_values.
    """

    def __init__(self, left, right):
        if False:
            print('Hello World!')
        super(MissingValueMismatch, self).__init__("LabelArray missing_values don't match: left={}, right={}".format(left, right))

class CategoryMismatch(ValueError):
    """
    Error raised on attempt to perform operations between LabelArrays with
    mismatched category arrays.
    """

    def __init__(self, left, right):
        if False:
            i = 10
            return i + 15
        (mismatches,) = np.where(left != right)
        assert len(mismatches), 'Not actually a mismatch!'
        super(CategoryMismatch, self).__init__("LabelArray categories don't match:\nMismatched Indices: {mismatches}\nLeft: {left}\nRight: {right}".format(mismatches=mismatches, left=left[mismatches], right=right[mismatches]))
_NotPassed = sentinel('_NotPassed')

class LabelArray(ndarray):
    """
    An ndarray subclass for working with arrays of strings.

    Factorizes the input array into integers, but overloads equality on strings
    to check against the factor label.

    Parameters
    ----------
    values : array-like
        Array of values that can be passed to np.asarray with dtype=object.
    missing_value : str
        Scalar value to treat as 'missing' for operations on ``self``.
    categories : list[str], optional
        List of values to use as categories.  If not supplied, categories will
        be inferred as the unique set of entries in ``values``.
    sort : bool, optional
        Whether to sort categories.  If sort is False and categories is
        supplied, they are left in the order provided.  If sort is False and
        categories is None, categories will be constructed in a random order.

    Attributes
    ----------
    categories : ndarray[str]
        An array containing the unique labels of self.
    reverse_categories : dict[str -> int]
        Reverse lookup table for ``categories``. Stores the index in
        ``categories`` at which each entry each unique entry is found.
    missing_value : str or None
        A sentinel missing value with NaN semantics for comparisons.

    Notes
    -----
    Consumers should be cautious when passing instances of LabelArray to numpy
    functions.  We attempt to disallow as many meaningless operations as
    possible, but since a LabelArray is just an ndarray of ints with some
    additional metadata, many numpy functions (for example, trigonometric) will
    happily accept a LabelArray and treat its values as though they were
    integers.

    In a future change, we may be able to disallow more numerical operations by
    creating a wrapper dtype which doesn't register an implementation for most
    numpy ufuncs. Until that change is made, consumers of LabelArray should
    assume that it is undefined behavior to pass a LabelArray to any numpy
    ufunc that operates on semantically-numerical data.

    See Also
    --------
    https://docs.scipy.org/doc/numpy-1.11.0/user/basics.subclassing.html
    """
    SUPPORTED_SCALAR_TYPES = (bytes, unicode, type(None))
    SUPPORTED_NON_NONE_SCALAR_TYPES = (bytes, unicode)

    @preprocess(values=coerce(list, partial(np.asarray, dtype=object)), categories=coerce((list, np.ndarray, set), list))
    @expect_types(values=np.ndarray, missing_value=SUPPORTED_SCALAR_TYPES, categories=optional(list))
    @expect_kinds(values=('O', 'S', 'U'))
    def __new__(cls, values, missing_value, categories=None, sort=True):
        if False:
            while True:
                i = 10
        if not is_object(values):
            values = values.astype(object)
        if values.flags.f_contiguous:
            ravel_order = 'F'
        else:
            ravel_order = 'C'
        if categories is None:
            (codes, categories, reverse_categories) = factorize_strings(values.ravel(ravel_order), missing_value=missing_value, sort=sort)
        else:
            (codes, categories, reverse_categories) = factorize_strings_known_categories(values.ravel(ravel_order), categories=categories, missing_value=missing_value, sort=sort)
        categories.setflags(write=False)
        return cls.from_codes_and_metadata(codes=codes.reshape(values.shape, order=ravel_order), categories=categories, reverse_categories=reverse_categories, missing_value=missing_value)

    @classmethod
    def from_codes_and_metadata(cls, codes, categories, reverse_categories, missing_value):
        if False:
            print('Hello World!')
        '\n        Rehydrate a LabelArray from the codes and metadata.\n\n        Parameters\n        ----------\n        codes : np.ndarray[integral]\n            The codes for the label array.\n        categories : np.ndarray[object]\n            The unique string categories.\n        reverse_categories : dict[str, int]\n            The mapping from category to its code-index.\n        missing_value : any\n            The value used to represent missing data.\n        '
        ret = codes.view(type=cls, dtype=np.void)
        ret._categories = categories
        ret._reverse_categories = reverse_categories
        ret._missing_value = missing_value
        return ret

    @classmethod
    def from_categorical(cls, categorical, missing_value=None):
        if False:
            print('Hello World!')
        '\n        Create a LabelArray from a pandas categorical.\n\n        Parameters\n        ----------\n        categorical : pd.Categorical\n            The categorical object to convert.\n        missing_value : bytes, unicode, or None, optional\n            The missing value to use for this LabelArray.\n\n        Returns\n        -------\n        la : LabelArray\n            The LabelArray representation of this categorical.\n        '
        return LabelArray(categorical, missing_value, categorical.categories)

    @property
    def categories(self):
        if False:
            i = 10
            return i + 15
        return self._categories

    @property
    def reverse_categories(self):
        if False:
            for i in range(10):
                print('nop')
        return self._reverse_categories

    @property
    def missing_value(self):
        if False:
            return 10
        return self._missing_value

    @property
    def missing_value_code(self):
        if False:
            i = 10
            return i + 15
        return self.reverse_categories[self.missing_value]

    def has_label(self, value):
        if False:
            print('Hello World!')
        return value in self.reverse_categories

    def __array_finalize__(self, obj):
        if False:
            while True:
                i = 10
        "\n        Called by Numpy after array construction.\n\n        There are three cases where this can happen:\n\n        1. Someone tries to directly construct a new array by doing::\n\n            >>> ndarray.__new__(LabelArray, ...)  # doctest: +SKIP\n\n           In this case, obj will be None.  We treat this as an error case and\n           fail.\n\n        2. Someone (most likely our own __new__) does::\n\n           >>> other_array.view(type=LabelArray)  # doctest: +SKIP\n\n           In this case, `self` will be the new LabelArray instance, and\n           ``obj` will be the array on which ``view`` is being called.\n\n           The caller of ``obj.view`` is responsible for setting category\n           metadata on ``self`` after we exit.\n\n        3. Someone creates a new LabelArray by slicing an existing one.\n\n           In this case, ``obj`` will be the original LabelArray.  We're\n           responsible for copying over the parent array's category metadata.\n        "
        if obj is None:
            raise TypeError('Direct construction of LabelArrays is not supported.')
        self._categories = getattr(obj, 'categories', None)
        self._reverse_categories = getattr(obj, 'reverse_categories', None)
        self._missing_value = getattr(obj, 'missing_value', None)

    def as_int_array(self):
        if False:
            print('Hello World!')
        '\n        Convert self into a regular ndarray of ints.\n\n        This is an O(1) operation. It does not copy the underlying data.\n        '
        return self.view(type=ndarray, dtype=unsigned_int_dtype_with_size_in_bytes(self.itemsize))

    def as_string_array(self):
        if False:
            i = 10
            return i + 15
        '\n        Convert self back into an array of strings.\n\n        This is an O(N) operation.\n        '
        return self.categories[self.as_int_array()]

    def as_categorical(self):
        if False:
            print('Hello World!')
        "\n        Coerce self into a pandas categorical.\n\n        This is only defined on 1D arrays, since that's all pandas supports.\n        "
        if len(self.shape) > 1:
            raise ValueError("Can't convert a 2D array to a categorical.")
        with ignore_pandas_nan_categorical_warning():
            return pd.Categorical.from_codes(self.as_int_array(), self.categories.copy(), ordered=False)

    def as_categorical_frame(self, index, columns, name=None):
        if False:
            i = 10
            return i + 15
        '\n        Coerce self into a pandas DataFrame of Categoricals.\n        '
        if len(self.shape) != 2:
            raise ValueError("Can't convert a non-2D LabelArray into a DataFrame.")
        expected_shape = (len(index), len(columns))
        if expected_shape != self.shape:
            raise ValueError("Can't construct a DataFrame with provided indices:\n\nLabelArray shape is {actual}, but index and columns imply that shape should be {expected}.".format(actual=self.shape, expected=expected_shape))
        return pd.Series(index=pd.MultiIndex.from_product([index, columns]), data=self.ravel().as_categorical(), name=name).unstack()

    def __setitem__(self, indexer, value):
        if False:
            while True:
                i = 10
        self_categories = self.categories
        if isinstance(value, self.SUPPORTED_SCALAR_TYPES):
            value_code = self.reverse_categories.get(value, None)
            if value_code is None:
                raise ValueError('%r is not in LabelArray categories.' % value)
            self.as_int_array()[indexer] = value_code
        elif isinstance(value, LabelArray):
            value_categories = value.categories
            if compare_arrays(self_categories, value_categories):
                return super(LabelArray, self).__setitem__(indexer, value)
            elif self.missing_value == value.missing_value and set(value.categories) <= set(self.categories):
                rhs = LabelArray.from_codes_and_metadata(*factorize_strings_known_categories(value.as_string_array().ravel(), list(self.categories), self.missing_value, False), missing_value=self.missing_value).reshape(value.shape)
                super(LabelArray, self).__setitem__(indexer, rhs)
            else:
                raise CategoryMismatch(self_categories, value_categories)
        else:
            raise NotImplementedError('Setting into a LabelArray with a value of type {type} is not yet supported.'.format(type=type(value).__name__))

    def set_scalar(self, indexer, value):
        if False:
            while True:
                i = 10
        '\n        Set scalar value into the array.\n\n        Parameters\n        ----------\n        indexer : any\n            The indexer to set the value at.\n        value : str\n            The value to assign at the given locations.\n\n        Raises\n        ------\n        ValueError\n            Raised when ``value`` is not a value element of this this label\n            array.\n        '
        try:
            value_code = self.reverse_categories[value]
        except KeyError:
            raise ValueError('%r is not in LabelArray categories.' % value)
        self.as_int_array()[indexer] = value_code

    def __setslice__(self, i, j, sequence):
        if False:
            while True:
                i = 10
        '\n        This method was deprecated in Python 2.0. It predates slice objects,\n        but Python 2.7.11 still uses it if you implement it, which ndarray\n        does.  In newer Pythons, __setitem__ is always called, but we need to\n        manuallly forward in py2.\n        '
        self.__setitem__(slice(i, j), sequence)

    def __getitem__(self, indexer):
        if False:
            print('Hello World!')
        result = super(LabelArray, self).__getitem__(indexer)
        if result.ndim:
            return result
        index = result.view(unsigned_int_dtype_with_size_in_bytes(self.itemsize))
        return self.categories[index]

    def is_missing(self):
        if False:
            i = 10
            return i + 15
        '\n        Like isnan, but checks for locations where we store missing values.\n        '
        return self.as_int_array() == self.reverse_categories[self.missing_value]

    def not_missing(self):
        if False:
            while True:
                i = 10
        '\n        Like ~isnan, but checks for locations where we store missing values.\n        '
        return self.as_int_array() != self.reverse_categories[self.missing_value]

    def _equality_check(op):
        if False:
            return 10
        '\n        Shared code for __eq__ and __ne__, parameterized on the actual\n        comparison operator to use.\n        '

        def method(self, other):
            if False:
                while True:
                    i = 10
            if isinstance(other, LabelArray):
                self_mv = self.missing_value
                other_mv = other.missing_value
                if self_mv != other_mv:
                    raise MissingValueMismatch(self_mv, other_mv)
                self_categories = self.categories
                other_categories = other.categories
                if not compare_arrays(self_categories, other_categories):
                    raise CategoryMismatch(self_categories, other_categories)
                return op(self.as_int_array(), other.as_int_array()) & self.not_missing() & other.not_missing()
            elif isinstance(other, ndarray):
                return op(self.as_string_array(), other) & self.not_missing()
            elif isinstance(other, self.SUPPORTED_SCALAR_TYPES):
                i = self._reverse_categories.get(other, -1)
                return op(self.as_int_array(), i) & self.not_missing()
            return op(super(LabelArray, self), other)
        return method
    __eq__ = _equality_check(eq)
    __ne__ = _equality_check(ne)
    del _equality_check

    def view(self, dtype=_NotPassed, type=_NotPassed):
        if False:
            i = 10
            return i + 15
        if type is _NotPassed and dtype not in (_NotPassed, self.dtype):
            raise TypeError("Can't view LabelArray as another dtype.")
        kwargs = {}
        if dtype is not _NotPassed:
            kwargs['dtype'] = dtype
        if type is not _NotPassed:
            kwargs['type'] = type
        return super(LabelArray, self).view(**kwargs)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        if False:
            while True:
                i = 10
        if dtype == self.dtype:
            if not subok:
                array = self.view(type=np.ndarray)
            else:
                array = self
            if copy:
                return array.copy()
            return array
        if dtype == object_dtype:
            return self.as_string_array()
        if dtype.kind == 'S':
            return self.as_string_array().astype(dtype, order=order, casting=casting, subok=subok, copy=copy)
        raise TypeError('%s can only be converted into object, string, or void, got: %r' % (type(self).__name__, dtype))
    SUPPORTED_NDARRAY_METHODS = frozenset(['astype', 'base', 'compress', 'copy', 'data', 'diagonal', 'dtype', 'flat', 'flatten', 'item', 'itemset', 'itemsize', 'nbytes', 'ndim', 'ravel', 'repeat', 'reshape', 'resize', 'setflags', 'shape', 'size', 'squeeze', 'strides', 'swapaxes', 'take', 'trace', 'transpose', 'view'])
    PUBLIC_NDARRAY_METHODS = frozenset([s for s in dir(ndarray) if not s.startswith('_')])
    locals().update({method: _make_unsupported_method(method) for method in PUBLIC_NDARRAY_METHODS - SUPPORTED_NDARRAY_METHODS})

    def __repr__(self):
        if False:
            print('Hello World!')
        repr_lines = repr(self.as_string_array()).splitlines()
        repr_lines[0] = repr_lines[0].replace('array(', 'LabelArray(', 1)
        repr_lines[-1] = repr_lines[-1].rsplit(',', 1)[0] + ')'
        return '\n     '.join(repr_lines)

    def empty_like(self, shape):
        if False:
            i = 10
            return i + 15
        '\n        Make an empty LabelArray with the same categories as ``self``, filled\n        with ``self.missing_value``.\n        '
        return type(self).from_codes_and_metadata(codes=np.full(shape, self.reverse_categories[self.missing_value], dtype=unsigned_int_dtype_with_size_in_bytes(self.itemsize)), categories=self.categories, reverse_categories=self.reverse_categories, missing_value=self.missing_value)

    def map_predicate(self, f):
        if False:
            print('Hello World!')
        '\n        Map a function from str -> bool element-wise over ``self``.\n\n        ``f`` will be applied exactly once to each non-missing unique value in\n        ``self``. Missing values will always return False.\n        '
        if self.missing_value is None:

            def f_to_use(x):
                if False:
                    for i in range(10):
                        print('nop')
                return False if x is None else f(x)
        else:
            f_to_use = f
        results = np.vectorize(f_to_use, otypes=[bool_dtype])(self.categories)
        results[self.reverse_categories[self.missing_value]] = False
        return results[self.as_int_array()]

    def map(self, f):
        if False:
            return 10
        '\n        Map a function from str -> str element-wise over ``self``.\n\n        ``f`` will be applied exactly once to each non-missing unique value in\n        ``self``. Missing values will always map to ``self.missing_value``.\n        '
        if self.missing_value is None:
            allowed_outtypes = self.SUPPORTED_SCALAR_TYPES
        else:
            allowed_outtypes = self.SUPPORTED_NON_NONE_SCALAR_TYPES

        def f_to_use(x, missing_value=self.missing_value, otypes=allowed_outtypes):
            if False:
                i = 10
                return i + 15
            if x == missing_value:
                return _sortable_sentinel
            ret = f(x)
            if not isinstance(ret, otypes):
                raise TypeError('LabelArray.map expected function {f} to return a string or None, but got {type} instead.\nValue was {value}.'.format(f=f.__name__, type=type(ret).__name__, value=ret))
            if ret == missing_value:
                return _sortable_sentinel
            return ret
        new_categories_with_duplicates = np.vectorize(f_to_use, otypes=[object])(self.categories)
        (new_categories, bloated_inverse_index) = np.unique(new_categories_with_duplicates, return_inverse=True)
        if new_categories[0] is _sortable_sentinel:
            new_categories[0] = self.missing_value
        reverse_index = bloated_inverse_index.astype(smallest_uint_that_can_hold(len(new_categories)))
        new_codes = np.take(reverse_index, self.as_int_array())
        return self.from_codes_and_metadata(new_codes, new_categories, dict(zip(new_categories, range(len(new_categories)))), missing_value=self.missing_value)

    def startswith(self, prefix):
        if False:
            while True:
                i = 10
        '\n        Element-wise startswith.\n\n        Parameters\n        ----------\n        prefix : str\n\n        Returns\n        -------\n        matches : np.ndarray[bool]\n            An array with the same shape as self indicating whether each\n            element of self started with ``prefix``.\n        '
        return self.map_predicate(lambda elem: elem.startswith(prefix))

    def endswith(self, suffix):
        if False:
            for i in range(10):
                print('nop')
        '\n        Elementwise endswith.\n\n        Parameters\n        ----------\n        suffix : str\n\n        Returns\n        -------\n        matches : np.ndarray[bool]\n            An array with the same shape as self indicating whether each\n            element of self ended with ``suffix``\n        '
        return self.map_predicate(lambda elem: elem.endswith(suffix))

    def has_substring(self, substring):
        if False:
            i = 10
            return i + 15
        '\n        Elementwise contains.\n\n        Parameters\n        ----------\n        substring : str\n\n        Returns\n        -------\n        matches : np.ndarray[bool]\n            An array with the same shape as self indicating whether each\n            element of self ended with ``suffix``.\n        '
        return self.map_predicate(lambda elem: substring in elem)

    @preprocess(pattern=coerce(from_=(bytes, unicode), to=re.compile))
    def matches(self, pattern):
        if False:
            i = 10
            return i + 15
        '\n        Elementwise regex match.\n\n        Parameters\n        ----------\n        pattern : str or compiled regex\n\n        Returns\n        -------\n        matches : np.ndarray[bool]\n            An array with the same shape as self indicating whether each\n            element of self was matched by ``pattern``.\n        '
        return self.map_predicate(compose(bool, pattern.match))

    @preprocess(container=coerce((list, tuple, np.ndarray), set))
    def element_of(self, container):
        if False:
            i = 10
            return i + 15
        '\n        Check if each element of self is an of ``container``.\n\n        Parameters\n        ----------\n        container : object\n            An object implementing a __contains__ to call on each element of\n            ``self``.\n\n        Returns\n        -------\n        is_contained : np.ndarray[bool]\n            An array with the same shape as self indicating whether each\n            element of self was an element of ``container``.\n        '
        return self.map_predicate(container.__contains__)

@instance
@total_ordering
class _sortable_sentinel(object):
    """Dummy object that sorts before any other python object.
    """

    def __eq__(self, other):
        if False:
            return 10
        return self is other

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        return True

@expect_types(trues=LabelArray, falses=LabelArray)
def labelarray_where(cond, trues, falses):
    if False:
        while True:
            i = 10
    'LabelArray-aware implementation of np.where.\n    '
    if trues.missing_value != falses.missing_value:
        raise ValueError("Can't compute where on arrays with different missing values.")
    strs = np.where(cond, trues.as_string_array(), falses.as_string_array())
    return LabelArray(strs, missing_value=trues.missing_value)