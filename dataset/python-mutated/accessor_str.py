from __future__ import annotations
import codecs
import re
import textwrap
from collections.abc import Hashable, Mapping
from functools import reduce
from operator import or_ as set_union
from re import Pattern
from typing import TYPE_CHECKING, Any, Callable, Generic
from unicodedata import normalize
import numpy as np
from xarray.core import duck_array_ops
from xarray.core.computation import apply_ufunc
from xarray.core.types import T_DataArray
if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from xarray.core.dataarray import DataArray
_cpython_optimized_encoders = ('utf-8', 'utf8', 'latin-1', 'latin1', 'iso-8859-1', 'mbcs', 'ascii')
_cpython_optimized_decoders = _cpython_optimized_encoders + ('utf-16', 'utf-32')

def _contains_obj_type(*, pat: Any, checker: Any) -> bool:
    if False:
        return 10
    'Determine if the object fits some rule or is array of objects that do so.'
    if isinstance(checker, type):
        targtype = checker
        checker = lambda x: isinstance(x, targtype)
    if checker(pat):
        return True
    if getattr(pat, 'dtype', 'no') != np.object_:
        return False
    return _apply_str_ufunc(func=checker, obj=pat).all()

def _contains_str_like(pat: Any) -> bool:
    if False:
        print('Hello World!')
    'Determine if the object is a str-like or array of str-like.'
    if isinstance(pat, (str, bytes)):
        return True
    if not hasattr(pat, 'dtype'):
        return False
    return pat.dtype.kind in ['U', 'S']

def _contains_compiled_re(pat: Any) -> bool:
    if False:
        print('Hello World!')
    'Determine if the object is a compiled re or array of compiled re.'
    return _contains_obj_type(pat=pat, checker=re.Pattern)

def _contains_callable(pat: Any) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Determine if the object is a callable or array of callables.'
    return _contains_obj_type(pat=pat, checker=callable)

def _apply_str_ufunc(*, func: Callable, obj: Any, dtype: DTypeLike=None, output_core_dims: list | tuple=((),), output_sizes: Mapping[Any, int] | None=None, func_args: tuple=(), func_kwargs: Mapping={}) -> Any:
    if False:
        for i in range(10):
            print('nop')
    if dtype is None:
        dtype = obj.dtype
    dask_gufunc_kwargs = dict()
    if output_sizes is not None:
        dask_gufunc_kwargs['output_sizes'] = output_sizes
    return apply_ufunc(func, obj, *func_args, vectorize=True, dask='parallelized', output_dtypes=[dtype], output_core_dims=output_core_dims, dask_gufunc_kwargs=dask_gufunc_kwargs, **func_kwargs)

class StringAccessor(Generic[T_DataArray]):
    """Vectorized string functions for string-like arrays.

    Similar to pandas, fields can be accessed through the `.str` attribute
    for applicable DataArrays.

        >>> da = xr.DataArray(["some", "text", "in", "an", "array"])
        >>> da.str.len()
        <xarray.DataArray (dim_0: 5)>
        array([4, 4, 2, 2, 5])
        Dimensions without coordinates: dim_0

    It also implements ``+``, ``*``, and ``%``, which operate as elementwise
    versions of the corresponding ``str`` methods. These will automatically
    broadcast for array-like inputs.

        >>> da1 = xr.DataArray(["first", "second", "third"], dims=["X"])
        >>> da2 = xr.DataArray([1, 2, 3], dims=["Y"])
        >>> da1.str + da2
        <xarray.DataArray (X: 3, Y: 3)>
        array([['first1', 'first2', 'first3'],
               ['second1', 'second2', 'second3'],
               ['third1', 'third2', 'third3']], dtype='<U7')
        Dimensions without coordinates: X, Y

        >>> da1 = xr.DataArray(["a", "b", "c", "d"], dims=["X"])
        >>> reps = xr.DataArray([3, 4], dims=["Y"])
        >>> da1.str * reps
        <xarray.DataArray (X: 4, Y: 2)>
        array([['aaa', 'aaaa'],
               ['bbb', 'bbbb'],
               ['ccc', 'cccc'],
               ['ddd', 'dddd']], dtype='<U4')
        Dimensions without coordinates: X, Y

        >>> da1 = xr.DataArray(["%s_%s", "%s-%s", "%s|%s"], dims=["X"])
        >>> da2 = xr.DataArray([1, 2], dims=["Y"])
        >>> da3 = xr.DataArray([0.1, 0.2], dims=["Z"])
        >>> da1.str % (da2, da3)
        <xarray.DataArray (X: 3, Y: 2, Z: 2)>
        array([[['1_0.1', '1_0.2'],
                ['2_0.1', '2_0.2']],
        <BLANKLINE>
               [['1-0.1', '1-0.2'],
                ['2-0.1', '2-0.2']],
        <BLANKLINE>
               [['1|0.1', '1|0.2'],
                ['2|0.1', '2|0.2']]], dtype='<U5')
        Dimensions without coordinates: X, Y, Z

    .. note::
        When using ``%`` formatting with a dict, the values are always used as a
        single value, they are not applied elementwise.

            >>> da1 = xr.DataArray(["%(a)s"], dims=["X"])
            >>> da2 = xr.DataArray([1, 2, 3], dims=["Y"])
            >>> da1 % {"a": da2}
            <xarray.DataArray (X: 1)>
            array(['<xarray.DataArray (Y: 3)>\\narray([1, 2, 3])\\nDimensions without coordinates: Y'],
                  dtype=object)
            Dimensions without coordinates: X
    """
    __slots__ = ('_obj',)

    def __init__(self, obj: T_DataArray) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._obj = obj

    def _stringify(self, invar: Any) -> str | bytes | Any:
        if False:
            print('Hello World!')
        '\n        Convert a string-like to the correct string/bytes type.\n\n        This is mostly here to tell mypy a pattern is a str/bytes not a re.Pattern.\n        '
        if hasattr(invar, 'astype'):
            return invar.astype(self._obj.dtype.kind)
        else:
            return self._obj.dtype.type(invar)

    def _apply(self, *, func: Callable, dtype: DTypeLike=None, output_core_dims: list | tuple=((),), output_sizes: Mapping[Any, int] | None=None, func_args: tuple=(), func_kwargs: Mapping={}) -> T_DataArray:
        if False:
            while True:
                i = 10
        return _apply_str_ufunc(obj=self._obj, func=func, dtype=dtype, output_core_dims=output_core_dims, output_sizes=output_sizes, func_args=func_args, func_kwargs=func_kwargs)

    def _re_compile(self, *, pat: str | bytes | Pattern | Any, flags: int=0, case: bool | None=None) -> Pattern | Any:
        if False:
            print('Hello World!')
        is_compiled_re = isinstance(pat, re.Pattern)
        if is_compiled_re and flags != 0:
            raise ValueError('Flags cannot be set when pat is a compiled regex.')
        if is_compiled_re and case is not None:
            raise ValueError('Case cannot be set when pat is a compiled regex.')
        if is_compiled_re:
            return re.compile(pat)
        if case is None:
            case = True
        if not case:
            flags |= re.IGNORECASE
        if getattr(pat, 'dtype', None) != np.object_:
            pat = self._stringify(pat)

        def func(x):
            if False:
                while True:
                    i = 10
            return re.compile(x, flags=flags)
        if isinstance(pat, np.ndarray):
            func_ = np.vectorize(func)
            return func_(pat)
        else:
            return _apply_str_ufunc(func=func, obj=pat, dtype=np.object_)

    def len(self) -> T_DataArray:
        if False:
            return 10
        '\n        Compute the length of each string in the array.\n\n        Returns\n        -------\n        lengths array : array of int\n        '
        return self._apply(func=len, dtype=int)

    def __getitem__(self, key: int | slice) -> T_DataArray:
        if False:
            return 10
        if isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self.get(key)

    def __add__(self, other: Any) -> T_DataArray:
        if False:
            while True:
                i = 10
        return self.cat(other, sep='')

    def __mul__(self, num: int | Any) -> T_DataArray:
        if False:
            for i in range(10):
                print('nop')
        return self.repeat(num)

    def __mod__(self, other: Any) -> T_DataArray:
        if False:
            while True:
                i = 10
        if isinstance(other, dict):
            other = {key: self._stringify(val) for (key, val) in other.items()}
            return self._apply(func=lambda x: x % other)
        elif isinstance(other, tuple):
            other = tuple((self._stringify(x) for x in other))
            return self._apply(func=lambda x, *y: x % y, func_args=other)
        else:
            return self._apply(func=lambda x, y: x % y, func_args=(other,))

    def get(self, i: int | Any, default: str | bytes='') -> T_DataArray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Extract character number `i` from each string in the array.\n\n        If `i` is array-like, they are broadcast against the array and\n        applied elementwise.\n\n        Parameters\n        ----------\n        i : int or array-like of int\n            Position of element to extract.\n            If array-like, it is broadcast.\n        default : str or bytes, default: ""\n            Value for out-of-range index.\n\n        Returns\n        -------\n        items : array of object\n        '

        def f(x, iind):
            if False:
                while True:
                    i = 10
            islice = slice(-1, None) if iind == -1 else slice(iind, iind + 1)
            item = x[islice]
            return item if item else default
        return self._apply(func=f, func_args=(i,))

    def slice(self, start: int | Any | None=None, stop: int | Any | None=None, step: int | Any | None=None) -> T_DataArray:
        if False:
            while True:
                i = 10
        "\n        Slice substrings from each string in the array.\n\n        If `start`, `stop`, or 'step` is array-like, they are broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        start : int or array-like of int, optional\n            Start position for slice operation.\n            If array-like, it is broadcast.\n        stop : int or array-like of int, optional\n            Stop position for slice operation.\n            If array-like, it is broadcast.\n        step : int or array-like of int, optional\n            Step size for slice operation.\n            If array-like, it is broadcast.\n\n        Returns\n        -------\n        sliced strings : same type as values\n        "
        f = lambda x, istart, istop, istep: x[slice(istart, istop, istep)]
        return self._apply(func=f, func_args=(start, stop, step))

    def slice_replace(self, start: int | Any | None=None, stop: int | Any | None=None, repl: str | bytes | Any='') -> T_DataArray:
        if False:
            i = 10
            return i + 15
        '\n        Replace a positional slice of a string with another value.\n\n        If `start`, `stop`, or \'repl` is array-like, they are broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        start : int or array-like of int, optional\n            Left index position to use for the slice. If not specified (None),\n            the slice is unbounded on the left, i.e. slice from the start\n            of the string. If array-like, it is broadcast.\n        stop : int or array-like of int, optional\n            Right index position to use for the slice. If not specified (None),\n            the slice is unbounded on the right, i.e. slice until the\n            end of the string. If array-like, it is broadcast.\n        repl : str or array-like of str, default: ""\n            String for replacement. If not specified, the sliced region\n            is replaced with an empty string. If array-like, it is broadcast.\n\n        Returns\n        -------\n        replaced : same type as values\n        '
        repl = self._stringify(repl)

        def func(x, istart, istop, irepl):
            if False:
                print('Hello World!')
            if len(x[istart:istop]) == 0:
                local_stop = istart
            else:
                local_stop = istop
            y = self._stringify('')
            if istart is not None:
                y += x[:istart]
            y += irepl
            if istop is not None:
                y += x[local_stop:]
            return y
        return self._apply(func=func, func_args=(start, stop, repl))

    def cat(self, *others, sep: str | bytes | Any='') -> T_DataArray:
        if False:
            i = 10
            return i + 15
        '\n        Concatenate strings elementwise in the DataArray with other strings.\n\n        The other strings can either be string scalars or other array-like.\n        Dimensions are automatically broadcast together.\n\n        An optional separator `sep` can also be specified. If `sep` is\n        array-like, it is broadcast against the array and applied elementwise.\n\n        Parameters\n        ----------\n        *others : str or array-like of str\n            Strings or array-like of strings to concatenate elementwise with\n            the current DataArray.\n        sep : str or array-like of str, default: "".\n            Separator to use between strings.\n            It is broadcast in the same way as the other input strings.\n            If array-like, its dimensions will be placed at the end of the output array dimensions.\n\n        Returns\n        -------\n        concatenated : same type as values\n\n        Examples\n        --------\n        Create a string array\n\n        >>> myarray = xr.DataArray(\n        ...     ["11111", "4"],\n        ...     dims=["X"],\n        ... )\n\n        Create some arrays to concatenate with it\n\n        >>> values_1 = xr.DataArray(\n        ...     ["a", "bb", "cccc"],\n        ...     dims=["Y"],\n        ... )\n        >>> values_2 = np.array(3.4)\n        >>> values_3 = ""\n        >>> values_4 = np.array("test", dtype=np.str_)\n\n        Determine the separator to use\n\n        >>> seps = xr.DataArray(\n        ...     [" ", ", "],\n        ...     dims=["ZZ"],\n        ... )\n\n        Concatenate the arrays using the separator\n\n        >>> myarray.str.cat(values_1, values_2, values_3, values_4, sep=seps)\n        <xarray.DataArray (X: 2, Y: 3, ZZ: 2)>\n        array([[[\'11111 a 3.4  test\', \'11111, a, 3.4, , test\'],\n                [\'11111 bb 3.4  test\', \'11111, bb, 3.4, , test\'],\n                [\'11111 cccc 3.4  test\', \'11111, cccc, 3.4, , test\']],\n        <BLANKLINE>\n               [[\'4 a 3.4  test\', \'4, a, 3.4, , test\'],\n                [\'4 bb 3.4  test\', \'4, bb, 3.4, , test\'],\n                [\'4 cccc 3.4  test\', \'4, cccc, 3.4, , test\']]], dtype=\'<U24\')\n        Dimensions without coordinates: X, Y, ZZ\n\n        See Also\n        --------\n        pandas.Series.str.cat\n        str.join\n        '
        sep = self._stringify(sep)
        others = tuple((self._stringify(x) for x in others))
        others = others + (sep,)
        func = lambda *x: x[-1].join(x[:-1])
        return self._apply(func=func, func_args=others, dtype=self._obj.dtype.kind)

    def join(self, dim: Hashable=None, sep: str | bytes | Any='') -> T_DataArray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Concatenate strings in a DataArray along a particular dimension.\n\n        An optional separator `sep` can also be specified. If `sep` is\n        array-like, it is broadcast against the array and applied elementwise.\n\n        Parameters\n        ----------\n        dim : hashable, optional\n            Dimension along which the strings should be concatenated.\n            Only one dimension is allowed at a time.\n            Optional for 0D or 1D DataArrays, required for multidimensional DataArrays.\n        sep : str or array-like, default: "".\n            Separator to use between strings.\n            It is broadcast in the same way as the other input strings.\n            If array-like, its dimensions will be placed at the end of the output array dimensions.\n\n        Returns\n        -------\n        joined : same type as values\n\n        Examples\n        --------\n        Create an array\n\n        >>> values = xr.DataArray(\n        ...     [["a", "bab", "abc"], ["abcd", "", "abcdef"]],\n        ...     dims=["X", "Y"],\n        ... )\n\n        Determine the separator\n\n        >>> seps = xr.DataArray(\n        ...     ["-", "_"],\n        ...     dims=["ZZ"],\n        ... )\n\n        Join the strings along a given dimension\n\n        >>> values.str.join(dim="Y", sep=seps)\n        <xarray.DataArray (X: 2, ZZ: 2)>\n        array([[\'a-bab-abc\', \'a_bab_abc\'],\n               [\'abcd--abcdef\', \'abcd__abcdef\']], dtype=\'<U12\')\n        Dimensions without coordinates: X, ZZ\n\n        See Also\n        --------\n        pandas.Series.str.join\n        str.join\n        '
        if self._obj.ndim > 1 and dim is None:
            raise ValueError('Dimension must be specified for multidimensional arrays.')
        if self._obj.ndim > 1:
            dimshifted = list(self._obj.transpose(dim, ...))
        elif self._obj.ndim == 1:
            dimshifted = list(self._obj)
        else:
            dimshifted = [self._obj]
        (start, *others) = dimshifted
        return start.str.cat(*others, sep=sep)

    def format(self, *args: Any, **kwargs: Any) -> T_DataArray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform python string formatting on each element of the DataArray.\n\n        This is equivalent to calling `str.format` on every element of the\n        DataArray. The replacement values can either be a string-like\n        scalar or array-like of string-like values. If array-like,\n        the values will be broadcast and applied elementwiseto the input\n        DataArray.\n\n        .. note::\n            Array-like values provided as `*args` will have their\n            dimensions added even if those arguments are not used in any\n            string formatting.\n\n        .. warning::\n            Array-like arguments are only applied elementwise for `*args`.\n            For `**kwargs`, values are used as-is.\n\n        Parameters\n        ----------\n        *args : str or bytes or array-like of str or bytes\n            Values for positional formatting.\n            If array-like, the values are broadcast and applied elementwise.\n            The dimensions will be placed at the end of the output array dimensions\n            in the order they are provided.\n        **kwargs : str or bytes or array-like of str or bytes\n            Values for keyword-based formatting.\n            These are **not** broadcast or applied elementwise.\n\n        Returns\n        -------\n        formatted : same type as values\n\n        Examples\n        --------\n        Create an array to format.\n\n        >>> values = xr.DataArray(\n        ...     ["{} is {adj0}", "{} and {} are {adj1}"],\n        ...     dims=["X"],\n        ... )\n\n        Set the values to fill.\n\n        >>> noun0 = xr.DataArray(\n        ...     ["spam", "egg"],\n        ...     dims=["Y"],\n        ... )\n        >>> noun1 = xr.DataArray(\n        ...     ["lancelot", "arthur"],\n        ...     dims=["ZZ"],\n        ... )\n        >>> adj0 = "unexpected"\n        >>> adj1 = "like a duck"\n\n        Insert the values into the array\n\n        >>> values.str.format(noun0, noun1, adj0=adj0, adj1=adj1)\n        <xarray.DataArray (X: 2, Y: 2, ZZ: 2)>\n        array([[[\'spam is unexpected\', \'spam is unexpected\'],\n                [\'egg is unexpected\', \'egg is unexpected\']],\n        <BLANKLINE>\n               [[\'spam and lancelot are like a duck\',\n                 \'spam and arthur are like a duck\'],\n                [\'egg and lancelot are like a duck\',\n                 \'egg and arthur are like a duck\']]], dtype=\'<U33\')\n        Dimensions without coordinates: X, Y, ZZ\n\n        See Also\n        --------\n        str.format\n        '
        args = tuple((self._stringify(x) for x in args))
        kwargs = {key: self._stringify(val) for (key, val) in kwargs.items()}
        func = lambda x, *args, **kwargs: self._obj.dtype.type.format(x, *args, **kwargs)
        return self._apply(func=func, func_args=args, func_kwargs={'kwargs': kwargs})

    def capitalize(self) -> T_DataArray:
        if False:
            i = 10
            return i + 15
        '\n        Convert strings in the array to be capitalized.\n\n        Returns\n        -------\n        capitalized : same type as values\n\n        Examples\n        --------\n        >>> da = xr.DataArray(\n        ...     ["temperature", "PRESSURE", "PreCipiTation", "daily rainfall"], dims="x"\n        ... )\n        >>> da\n        <xarray.DataArray (x: 4)>\n        array([\'temperature\', \'PRESSURE\', \'PreCipiTation\', \'daily rainfall\'],\n              dtype=\'<U14\')\n        Dimensions without coordinates: x\n        >>> capitalized = da.str.capitalize()\n        >>> capitalized\n        <xarray.DataArray (x: 4)>\n        array([\'Temperature\', \'Pressure\', \'Precipitation\', \'Daily rainfall\'],\n              dtype=\'<U14\')\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.capitalize())

    def lower(self) -> T_DataArray:
        if False:
            i = 10
            return i + 15
        '\n        Convert strings in the array to lowercase.\n\n        Returns\n        -------\n        lowered : same type as values\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["Temperature", "PRESSURE"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 2)>\n        array([\'Temperature\', \'PRESSURE\'], dtype=\'<U11\')\n        Dimensions without coordinates: x\n        >>> lowered = da.str.lower()\n        >>> lowered\n        <xarray.DataArray (x: 2)>\n        array([\'temperature\', \'pressure\'], dtype=\'<U11\')\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.lower())

    def swapcase(self) -> T_DataArray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert strings in the array to be swapcased.\n\n        Returns\n        -------\n        swapcased : same type as values\n\n        Examples\n        --------\n        >>> import xarray as xr\n        >>> da = xr.DataArray(["temperature", "PRESSURE", "HuMiDiTy"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 3)>\n        array([\'temperature\', \'PRESSURE\', \'HuMiDiTy\'], dtype=\'<U11\')\n        Dimensions without coordinates: x\n        >>> swapcased = da.str.swapcase()\n        >>> swapcased\n        <xarray.DataArray (x: 3)>\n        array([\'TEMPERATURE\', \'pressure\', \'hUmIdItY\'], dtype=\'<U11\')\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.swapcase())

    def title(self) -> T_DataArray:
        if False:
            i = 10
            return i + 15
        '\n        Convert strings in the array to titlecase.\n\n        Returns\n        -------\n        titled : same type as values\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["temperature", "PRESSURE", "HuMiDiTy"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 3)>\n        array([\'temperature\', \'PRESSURE\', \'HuMiDiTy\'], dtype=\'<U11\')\n        Dimensions without coordinates: x\n        >>> titled = da.str.title()\n        >>> titled\n        <xarray.DataArray (x: 3)>\n        array([\'Temperature\', \'Pressure\', \'Humidity\'], dtype=\'<U11\')\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.title())

    def upper(self) -> T_DataArray:
        if False:
            print('Hello World!')
        '\n        Convert strings in the array to uppercase.\n\n        Returns\n        -------\n        uppered : same type as values\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["temperature", "HuMiDiTy"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 2)>\n        array([\'temperature\', \'HuMiDiTy\'], dtype=\'<U11\')\n        Dimensions without coordinates: x\n        >>> uppered = da.str.upper()\n        >>> uppered\n        <xarray.DataArray (x: 2)>\n        array([\'TEMPERATURE\', \'HUMIDITY\'], dtype=\'<U11\')\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.upper())

    def casefold(self) -> T_DataArray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert strings in the array to be casefolded.\n\n        Casefolding is similar to converting to lowercase,\n        but removes all case distinctions.\n        This is important in some languages that have more complicated\n        cases and case conversions. For example,\n        the \'ß\' character in German is case-folded to \'ss\', whereas it is lowercased\n        to \'ß\'.\n\n        Returns\n        -------\n        casefolded : same type as values\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["TEMPERATURE", "HuMiDiTy"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 2)>\n        array([\'TEMPERATURE\', \'HuMiDiTy\'], dtype=\'<U11\')\n        Dimensions without coordinates: x\n        >>> casefolded = da.str.casefold()\n        >>> casefolded\n        <xarray.DataArray (x: 2)>\n        array([\'temperature\', \'humidity\'], dtype=\'<U11\')\n        Dimensions without coordinates: x\n\n        >>> da = xr.DataArray(["ß", "İ"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 2)>\n        array([\'ß\', \'İ\'], dtype=\'<U1\')\n        Dimensions without coordinates: x\n        >>> casefolded = da.str.casefold()\n        >>> casefolded\n        <xarray.DataArray (x: 2)>\n        array([\'ss\', \'i̇\'], dtype=\'<U2\')\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.casefold())

    def normalize(self, form: str) -> T_DataArray:
        if False:
            print('Hello World!')
        '\n        Return the Unicode normal form for the strings in the datarray.\n\n        For more information on the forms, see the documentation for\n        :func:`unicodedata.normalize`.\n\n        Parameters\n        ----------\n        form : {"NFC", "NFKC", "NFD", "NFKD"}\n            Unicode form.\n\n        Returns\n        -------\n        normalized : same type as values\n\n        '
        return self._apply(func=lambda x: normalize(form, x))

    def isalnum(self) -> T_DataArray:
        if False:
            return 10
        '\n        Check whether all characters in each string are alphanumeric.\n\n        Returns\n        -------\n        isalnum : array of bool\n            Array of boolean values with the same shape as the original array.\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["H2O", "NaCl-"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 2)>\n        array([\'H2O\', \'NaCl-\'], dtype=\'<U5\')\n        Dimensions without coordinates: x\n        >>> isalnum = da.str.isalnum()\n        >>> isalnum\n        <xarray.DataArray (x: 2)>\n        array([ True, False])\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.isalnum(), dtype=bool)

    def isalpha(self) -> T_DataArray:
        if False:
            print('Hello World!')
        '\n        Check whether all characters in each string are alphabetic.\n\n        Returns\n        -------\n        isalpha : array of bool\n            Array of boolean values with the same shape as the original array.\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["Mn", "H2O", "NaCl-"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 3)>\n        array([\'Mn\', \'H2O\', \'NaCl-\'], dtype=\'<U5\')\n        Dimensions without coordinates: x\n        >>> isalpha = da.str.isalpha()\n        >>> isalpha\n        <xarray.DataArray (x: 3)>\n        array([ True, False, False])\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.isalpha(), dtype=bool)

    def isdecimal(self) -> T_DataArray:
        if False:
            i = 10
            return i + 15
        '\n        Check whether all characters in each string are decimal.\n\n        Returns\n        -------\n        isdecimal : array of bool\n            Array of boolean values with the same shape as the original array.\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["2.3", "123", "0"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 3)>\n        array([\'2.3\', \'123\', \'0\'], dtype=\'<U3\')\n        Dimensions without coordinates: x\n        >>> isdecimal = da.str.isdecimal()\n        >>> isdecimal\n        <xarray.DataArray (x: 3)>\n        array([False,  True,  True])\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.isdecimal(), dtype=bool)

    def isdigit(self) -> T_DataArray:
        if False:
            while True:
                i = 10
        '\n        Check whether all characters in each string are digits.\n\n        Returns\n        -------\n        isdigit : array of bool\n            Array of boolean values with the same shape as the original array.\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["123", "1.2", "0", "CO2", "NaCl"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 5)>\n        array([\'123\', \'1.2\', \'0\', \'CO2\', \'NaCl\'], dtype=\'<U4\')\n        Dimensions without coordinates: x\n        >>> isdigit = da.str.isdigit()\n        >>> isdigit\n        <xarray.DataArray (x: 5)>\n        array([ True, False,  True, False, False])\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.isdigit(), dtype=bool)

    def islower(self) -> T_DataArray:
        if False:
            while True:
                i = 10
        '\n        Check whether all characters in each string are lowercase.\n\n        Returns\n        -------\n        islower : array of bool\n            Array of boolean values with the same shape as the original array indicating whether all characters of each\n            element of the string array are lowercase (True) or not (False).\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["temperature", "HUMIDITY", "pREciPiTaTioN"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 3)>\n        array([\'temperature\', \'HUMIDITY\', \'pREciPiTaTioN\'], dtype=\'<U13\')\n        Dimensions without coordinates: x\n        >>> islower = da.str.islower()\n        >>> islower\n        <xarray.DataArray (x: 3)>\n        array([ True, False, False])\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.islower(), dtype=bool)

    def isnumeric(self) -> T_DataArray:
        if False:
            return 10
        '\n        Check whether all characters in each string are numeric.\n\n        Returns\n        -------\n        isnumeric : array of bool\n            Array of boolean values with the same shape as the original array.\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["123", "2.3", "H2O", "NaCl-", "Mn"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 5)>\n        array([\'123\', \'2.3\', \'H2O\', \'NaCl-\', \'Mn\'], dtype=\'<U5\')\n        Dimensions without coordinates: x\n        >>> isnumeric = da.str.isnumeric()\n        >>> isnumeric\n        <xarray.DataArray (x: 5)>\n        array([ True, False, False, False, False])\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.isnumeric(), dtype=bool)

    def isspace(self) -> T_DataArray:
        if False:
            while True:
                i = 10
        '\n        Check whether all characters in each string are spaces.\n\n        Returns\n        -------\n        isspace : array of bool\n            Array of boolean values with the same shape as the original array.\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["", " ", "\\t", "\\n"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 4)>\n        array([\'\', \' \', \'\\t\', \'\\n\'], dtype=\'<U1\')\n        Dimensions without coordinates: x\n        >>> isspace = da.str.isspace()\n        >>> isspace\n        <xarray.DataArray (x: 4)>\n        array([False,  True,  True,  True])\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.isspace(), dtype=bool)

    def istitle(self) -> T_DataArray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Check whether all characters in each string are titlecase.\n\n        Returns\n        -------\n        istitle : array of bool\n            Array of boolean values with the same shape as the original array.\n\n        Examples\n        --------\n        >>> da = xr.DataArray(\n        ...     [\n        ...         "The Evolution Of Species",\n        ...         "The Theory of relativity",\n        ...         "the quantum mechanics of atoms",\n        ...     ],\n        ...     dims="title",\n        ... )\n        >>> da\n        <xarray.DataArray (title: 3)>\n        array([\'The Evolution Of Species\', \'The Theory of relativity\',\n               \'the quantum mechanics of atoms\'], dtype=\'<U30\')\n        Dimensions without coordinates: title\n        >>> istitle = da.str.istitle()\n        >>> istitle\n        <xarray.DataArray (title: 3)>\n        array([ True, False, False])\n        Dimensions without coordinates: title\n        '
        return self._apply(func=lambda x: x.istitle(), dtype=bool)

    def isupper(self) -> T_DataArray:
        if False:
            return 10
        '\n        Check whether all characters in each string are uppercase.\n\n        Returns\n        -------\n        isupper : array of bool\n            Array of boolean values with the same shape as the original array.\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["TEMPERATURE", "humidity", "PreCIpiTAtioN"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 3)>\n        array([\'TEMPERATURE\', \'humidity\', \'PreCIpiTAtioN\'], dtype=\'<U13\')\n        Dimensions without coordinates: x\n        >>> isupper = da.str.isupper()\n        >>> isupper\n        <xarray.DataArray (x: 3)>\n        array([ True, False, False])\n        Dimensions without coordinates: x\n        '
        return self._apply(func=lambda x: x.isupper(), dtype=bool)

    def count(self, pat: str | bytes | Pattern | Any, flags: int=0, case: bool | None=None) -> T_DataArray:
        if False:
            print('Hello World!')
        '\n        Count occurrences of pattern in each string of the array.\n\n        This function is used to count the number of times a particular regex\n        pattern is repeated in each of the string elements of the\n        :class:`~xarray.DataArray`.\n\n        The pattern `pat` can either be a single ``str`` or `re.Pattern` or\n        array-like of ``str`` or `re.Pattern`. If array-like, it is broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        pat : str or re.Pattern or array-like of str or re.Pattern\n            A string containing a regular expression or a compiled regular\n            expression object. If array-like, it is broadcast.\n        flags : int, default: 0\n            Flags to pass through to the re module, e.g. `re.IGNORECASE`.\n            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.\n            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.\n            Cannot be set if `pat` is a compiled regex.\n        case : bool, default: True\n            If True, case sensitive.\n            Cannot be set if `pat` is a compiled regex.\n            Equivalent to setting the `re.IGNORECASE` flag.\n\n        Returns\n        -------\n        counts : array of int\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["jjklmn", "opjjqrs", "t-JJ99vwx"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 3)>\n        array([\'jjklmn\', \'opjjqrs\', \'t-JJ99vwx\'], dtype=\'<U9\')\n        Dimensions without coordinates: x\n\n        Using a string:\n        >>> da.str.count("jj")\n        <xarray.DataArray (x: 3)>\n        array([1, 1, 0])\n        Dimensions without coordinates: x\n\n        Enable case-insensitive matching by setting case to false:\n        >>> counts = da.str.count("jj", case=False)\n        >>> counts\n        <xarray.DataArray (x: 3)>\n        array([1, 1, 1])\n        Dimensions without coordinates: x\n\n        Using regex:\n        >>> pat = "JJ[0-9]{2}[a-z]{3}"\n        >>> counts = da.str.count(pat)\n        >>> counts\n        <xarray.DataArray (x: 3)>\n        array([0, 0, 1])\n        Dimensions without coordinates: x\n\n        Using an array of strings (the pattern will be broadcast against the array):\n\n        >>> pat = xr.DataArray(["jj", "JJ"], dims="y")\n        >>> counts = da.str.count(pat)\n        >>> counts\n        <xarray.DataArray (x: 3, y: 2)>\n        array([[1, 0],\n               [1, 0],\n               [0, 1]])\n        Dimensions without coordinates: x, y\n        '
        pat = self._re_compile(pat=pat, flags=flags, case=case)
        func = lambda x, ipat: len(ipat.findall(x))
        return self._apply(func=func, func_args=(pat,), dtype=int)

    def startswith(self, pat: str | bytes | Any) -> T_DataArray:
        if False:
            return 10
        '\n        Test if the start of each string in the array matches a pattern.\n\n        The pattern `pat` can either be a ``str`` or array-like of ``str``.\n        If array-like, it will be broadcast and applied elementwise.\n\n        Parameters\n        ----------\n        pat : str\n            Character sequence. Regular expressions are not accepted.\n            If array-like, it is broadcast.\n\n        Returns\n        -------\n        startswith : array of bool\n            An array of booleans indicating whether the given pattern matches\n            the start of each string element.\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["$100", "£23", "100"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 3)>\n        array([\'$100\', \'£23\', \'100\'], dtype=\'<U4\')\n        Dimensions without coordinates: x\n        >>> startswith = da.str.startswith("$")\n        >>> startswith\n        <xarray.DataArray (x: 3)>\n        array([ True, False, False])\n        Dimensions without coordinates: x\n        '
        pat = self._stringify(pat)
        func = lambda x, y: x.startswith(y)
        return self._apply(func=func, func_args=(pat,), dtype=bool)

    def endswith(self, pat: str | bytes | Any) -> T_DataArray:
        if False:
            i = 10
            return i + 15
        '\n        Test if the end of each string in the array matches a pattern.\n\n        The pattern `pat` can either be a ``str`` or array-like of ``str``.\n        If array-like, it will be broadcast and applied elementwise.\n\n        Parameters\n        ----------\n        pat : str\n            Character sequence. Regular expressions are not accepted.\n            If array-like, it is broadcast.\n\n        Returns\n        -------\n        endswith : array of bool\n            A Series of booleans indicating whether the given pattern matches\n            the end of each string element.\n\n        Examples\n        --------\n        >>> da = xr.DataArray(["10C", "10c", "100F"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 3)>\n        array([\'10C\', \'10c\', \'100F\'], dtype=\'<U4\')\n        Dimensions without coordinates: x\n        >>> endswith = da.str.endswith("C")\n        >>> endswith\n        <xarray.DataArray (x: 3)>\n        array([ True, False, False])\n        Dimensions without coordinates: x\n        '
        pat = self._stringify(pat)
        func = lambda x, y: x.endswith(y)
        return self._apply(func=func, func_args=(pat,), dtype=bool)

    def pad(self, width: int | Any, side: str='left', fillchar: str | bytes | Any=' ') -> T_DataArray:
        if False:
            print('Hello World!')
        '\n        Pad strings in the array up to width.\n\n        If `width` or \'fillchar` is array-like, they are broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        width : int or array-like of int\n            Minimum width of resulting string; additional characters will be\n            filled with character defined in ``fillchar``.\n            If array-like, it is broadcast.\n        side : {"left", "right", "both"}, default: "left"\n            Side from which to fill resulting string.\n        fillchar : str or array-like of str, default: " "\n            Additional character for filling, default is a space.\n            If array-like, it is broadcast.\n\n        Returns\n        -------\n        filled : same type as values\n            Array with a minimum number of char in each element.\n\n        Examples\n        --------\n        Pad strings in the array with a single string on the left side.\n\n        Define the string in the array.\n\n        >>> da = xr.DataArray(["PAR184", "TKO65", "NBO9139", "NZ39"], dims="x")\n        >>> da\n        <xarray.DataArray (x: 4)>\n        array([\'PAR184\', \'TKO65\', \'NBO9139\', \'NZ39\'], dtype=\'<U7\')\n        Dimensions without coordinates: x\n\n        Pad the strings\n\n        >>> filled = da.str.pad(8, side="left", fillchar="0")\n        >>> filled\n        <xarray.DataArray (x: 4)>\n        array([\'00PAR184\', \'000TKO65\', \'0NBO9139\', \'0000NZ39\'], dtype=\'<U8\')\n        Dimensions without coordinates: x\n\n        Pad strings on the right side\n\n        >>> filled = da.str.pad(8, side="right", fillchar="0")\n        >>> filled\n        <xarray.DataArray (x: 4)>\n        array([\'PAR18400\', \'TKO65000\', \'NBO91390\', \'NZ390000\'], dtype=\'<U8\')\n        Dimensions without coordinates: x\n\n        Pad strings on both sides\n\n        >>> filled = da.str.pad(8, side="both", fillchar="0")\n        >>> filled\n        <xarray.DataArray (x: 4)>\n        array([\'0PAR1840\', \'0TKO6500\', \'NBO91390\', \'00NZ3900\'], dtype=\'<U8\')\n        Dimensions without coordinates: x\n\n        Using an array-like width\n\n        >>> width = xr.DataArray([8, 10], dims="y")\n        >>> filled = da.str.pad(width, side="left", fillchar="0")\n        >>> filled\n        <xarray.DataArray (x: 4, y: 2)>\n        array([[\'00PAR184\', \'0000PAR184\'],\n               [\'000TKO65\', \'00000TKO65\'],\n               [\'0NBO9139\', \'000NBO9139\'],\n               [\'0000NZ39\', \'000000NZ39\']], dtype=\'<U10\')\n        Dimensions without coordinates: x, y\n\n        Using an array-like value for fillchar\n\n        >>> fillchar = xr.DataArray(["0", "-"], dims="y")\n        >>> filled = da.str.pad(8, side="left", fillchar=fillchar)\n        >>> filled\n        <xarray.DataArray (x: 4, y: 2)>\n        array([[\'00PAR184\', \'--PAR184\'],\n               [\'000TKO65\', \'---TKO65\'],\n               [\'0NBO9139\', \'-NBO9139\'],\n               [\'0000NZ39\', \'----NZ39\']], dtype=\'<U8\')\n        Dimensions without coordinates: x, y\n        '
        if side == 'left':
            func = self.rjust
        elif side == 'right':
            func = self.ljust
        elif side == 'both':
            func = self.center
        else:
            raise ValueError('Invalid side')
        return func(width=width, fillchar=fillchar)

    def _padder(self, *, func: Callable, width: int | Any, fillchar: str | bytes | Any=' ') -> T_DataArray:
        if False:
            i = 10
            return i + 15
        '\n        Wrapper function to handle padding operations\n        '
        fillchar = self._stringify(fillchar)

        def overfunc(x, iwidth, ifillchar):
            if False:
                while True:
                    i = 10
            if len(ifillchar) != 1:
                raise TypeError('fillchar must be a character, not str')
            return func(x, int(iwidth), ifillchar)
        return self._apply(func=overfunc, func_args=(width, fillchar))

    def center(self, width: int | Any, fillchar: str | bytes | Any=' ') -> T_DataArray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Pad left and right side of each string in the array.\n\n        If `width` or \'fillchar` is array-like, they are broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        width : int or array-like of int\n            Minimum width of resulting string; additional characters will be\n            filled with ``fillchar``. If array-like, it is broadcast.\n        fillchar : str or array-like of str, default: " "\n            Additional character for filling, default is a space.\n            If array-like, it is broadcast.\n\n        Returns\n        -------\n        filled : same type as values\n        '
        func = self._obj.dtype.type.center
        return self._padder(func=func, width=width, fillchar=fillchar)

    def ljust(self, width: int | Any, fillchar: str | bytes | Any=' ') -> T_DataArray:
        if False:
            print('Hello World!')
        '\n        Pad right side of each string in the array.\n\n        If `width` or \'fillchar` is array-like, they are broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        width : int or array-like of int\n            Minimum width of resulting string; additional characters will be\n            filled with ``fillchar``. If array-like, it is broadcast.\n        fillchar : str or array-like of str, default: " "\n            Additional character for filling, default is a space.\n            If array-like, it is broadcast.\n\n        Returns\n        -------\n        filled : same type as values\n        '
        func = self._obj.dtype.type.ljust
        return self._padder(func=func, width=width, fillchar=fillchar)

    def rjust(self, width: int | Any, fillchar: str | bytes | Any=' ') -> T_DataArray:
        if False:
            i = 10
            return i + 15
        '\n        Pad left side of each string in the array.\n\n        If `width` or \'fillchar` is array-like, they are broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        width : int or array-like of int\n            Minimum width of resulting string; additional characters will be\n            filled with ``fillchar``. If array-like, it is broadcast.\n        fillchar : str or array-like of str, default: " "\n            Additional character for filling, default is a space.\n            If array-like, it is broadcast.\n\n        Returns\n        -------\n        filled : same type as values\n        '
        func = self._obj.dtype.type.rjust
        return self._padder(func=func, width=width, fillchar=fillchar)

    def zfill(self, width: int | Any) -> T_DataArray:
        if False:
            while True:
                i = 10
        "\n        Pad each string in the array by prepending '0' characters.\n\n        Strings in the array are padded with '0' characters on the\n        left of the string to reach a total string length  `width`. Strings\n        in the array with length greater or equal to `width` are unchanged.\n\n        If `width` is array-like, it is broadcast against the array and applied\n        elementwise.\n\n        Parameters\n        ----------\n        width : int or array-like of int\n            Minimum length of resulting string; strings with length less\n            than `width` be prepended with '0' characters. If array-like, it is broadcast.\n\n        Returns\n        -------\n        filled : same type as values\n        "
        return self.rjust(width, fillchar='0')

    def contains(self, pat: str | bytes | Pattern | Any, case: bool | None=None, flags: int=0, regex: bool=True) -> T_DataArray:
        if False:
            while True:
                i = 10
        '\n        Test if pattern or regex is contained within each string of the array.\n\n        Return boolean array based on whether a given pattern or regex is\n        contained within a string of the array.\n\n        The pattern `pat` can either be a single ``str`` or `re.Pattern` or\n        array-like of ``str`` or `re.Pattern`. If array-like, it is broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        pat : str or re.Pattern or array-like of str or re.Pattern\n            Character sequence, a string containing a regular expression,\n            or a compiled regular expression object. If array-like, it is broadcast.\n        case : bool, default: True\n            If True, case sensitive.\n            Cannot be set if `pat` is a compiled regex.\n            Equivalent to setting the `re.IGNORECASE` flag.\n        flags : int, default: 0\n            Flags to pass through to the re module, e.g. `re.IGNORECASE`.\n            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.\n            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.\n            Cannot be set if `pat` is a compiled regex.\n        regex : bool, default: True\n            If True, assumes the pat is a regular expression.\n            If False, treats the pat as a literal string.\n            Cannot be set to `False` if `pat` is a compiled regex.\n\n        Returns\n        -------\n        contains : array of bool\n            An array of boolean values indicating whether the\n            given pattern is contained within the string of each element\n            of the array.\n        '
        is_compiled_re = _contains_compiled_re(pat)
        if is_compiled_re and (not regex):
            raise ValueError('Must use regular expression matching for regular expression object.')
        if regex:
            if not is_compiled_re:
                pat = self._re_compile(pat=pat, flags=flags, case=case)

            def func(x, ipat):
                if False:
                    while True:
                        i = 10
                if ipat.groups > 0:
                    raise ValueError('This pattern has match groups.')
                return bool(ipat.search(x))
        else:
            pat = self._stringify(pat)
            if case or case is None:
                func = lambda x, ipat: ipat in x
            elif self._obj.dtype.char == 'U':
                uppered = self.casefold()
                uppat = StringAccessor(pat).casefold()
                return uppered.str.contains(uppat, regex=False)
            else:
                uppered = self.upper()
                uppat = StringAccessor(pat).upper()
                return uppered.str.contains(uppat, regex=False)
        return self._apply(func=func, func_args=(pat,), dtype=bool)

    def match(self, pat: str | bytes | Pattern | Any, case: bool | None=None, flags: int=0) -> T_DataArray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine if each string in the array matches a regular expression.\n\n        The pattern `pat` can either be a single ``str`` or `re.Pattern` or\n        array-like of ``str`` or `re.Pattern`. If array-like, it is broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        pat : str or re.Pattern or array-like of str or re.Pattern\n            A string containing a regular expression or\n            a compiled regular expression object. If array-like, it is broadcast.\n        case : bool, default: True\n            If True, case sensitive.\n            Cannot be set if `pat` is a compiled regex.\n            Equivalent to setting the `re.IGNORECASE` flag.\n        flags : int, default: 0\n            Flags to pass through to the re module, e.g. `re.IGNORECASE`.\n            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.\n            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.\n            Cannot be set if `pat` is a compiled regex.\n\n        Returns\n        -------\n        matched : array of bool\n        '
        pat = self._re_compile(pat=pat, flags=flags, case=case)
        func = lambda x, ipat: bool(ipat.match(x))
        return self._apply(func=func, func_args=(pat,), dtype=bool)

    def strip(self, to_strip: str | bytes | Any=None, side: str='both') -> T_DataArray:
        if False:
            print('Hello World!')
        '\n        Remove leading and trailing characters.\n\n        Strip whitespaces (including newlines) or a set of specified characters\n        from each string in the array from left and/or right sides.\n\n        `to_strip` can either be a ``str`` or array-like of ``str``.\n        If array-like, it will be broadcast and applied elementwise.\n\n        Parameters\n        ----------\n        to_strip : str or array-like of str or None, default: None\n            Specifying the set of characters to be removed.\n            All combinations of this set of characters will be stripped.\n            If None then whitespaces are removed. If array-like, it is broadcast.\n        side : {"left", "right", "both"}, default: "both"\n            Side from which to strip.\n\n        Returns\n        -------\n        stripped : same type as values\n        '
        if to_strip is not None:
            to_strip = self._stringify(to_strip)
        if side == 'both':
            func = lambda x, y: x.strip(y)
        elif side == 'left':
            func = lambda x, y: x.lstrip(y)
        elif side == 'right':
            func = lambda x, y: x.rstrip(y)
        else:
            raise ValueError('Invalid side')
        return self._apply(func=func, func_args=(to_strip,))

    def lstrip(self, to_strip: str | bytes | Any=None) -> T_DataArray:
        if False:
            print('Hello World!')
        '\n        Remove leading characters.\n\n        Strip whitespaces (including newlines) or a set of specified characters\n        from each string in the array from the left side.\n\n        `to_strip` can either be a ``str`` or array-like of ``str``.\n        If array-like, it will be broadcast and applied elementwise.\n\n        Parameters\n        ----------\n        to_strip : str or array-like of str or None, default: None\n            Specifying the set of characters to be removed.\n            All combinations of this set of characters will be stripped.\n            If None then whitespaces are removed. If array-like, it is broadcast.\n\n        Returns\n        -------\n        stripped : same type as values\n        '
        return self.strip(to_strip, side='left')

    def rstrip(self, to_strip: str | bytes | Any=None) -> T_DataArray:
        if False:
            while True:
                i = 10
        '\n        Remove trailing characters.\n\n        Strip whitespaces (including newlines) or a set of specified characters\n        from each string in the array from the right side.\n\n        `to_strip` can either be a ``str`` or array-like of ``str``.\n        If array-like, it will be broadcast and applied elementwise.\n\n        Parameters\n        ----------\n        to_strip : str or array-like of str or None, default: None\n            Specifying the set of characters to be removed.\n            All combinations of this set of characters will be stripped.\n            If None then whitespaces are removed. If array-like, it is broadcast.\n\n        Returns\n        -------\n        stripped : same type as values\n        '
        return self.strip(to_strip, side='right')

    def wrap(self, width: int | Any, **kwargs) -> T_DataArray:
        if False:
            return 10
        '\n        Wrap long strings in the array in paragraphs with length less than `width`.\n\n        This method has the same keyword parameters and defaults as\n        :class:`textwrap.TextWrapper`.\n\n        If `width` is array-like, it is broadcast against the array and applied\n        elementwise.\n\n        Parameters\n        ----------\n        width : int or array-like of int\n            Maximum line-width.\n            If array-like, it is broadcast.\n        **kwargs\n            keyword arguments passed into :class:`textwrap.TextWrapper`.\n\n        Returns\n        -------\n        wrapped : same type as values\n        '
        ifunc = lambda x: textwrap.TextWrapper(width=x, **kwargs)
        tw = StringAccessor(width)._apply(func=ifunc, dtype=np.object_)
        func = lambda x, itw: '\n'.join(itw.wrap(x))
        return self._apply(func=func, func_args=(tw,))

    def translate(self, table: Mapping[Any, str | bytes | int | None]) -> T_DataArray:
        if False:
            print('Hello World!')
        '\n        Map characters of each string through the given mapping table.\n\n        Parameters\n        ----------\n        table : dict-like from and to str or bytes or int\n            A a mapping of Unicode ordinals to Unicode ordinals, strings, int\n            or None. Unmapped characters are left untouched. Characters mapped\n            to None are deleted. :meth:`str.maketrans` is a helper function for\n            making translation tables.\n\n        Returns\n        -------\n        translated : same type as values\n        '
        func = lambda x: x.translate(table)
        return self._apply(func=func)

    def repeat(self, repeats: int | Any) -> T_DataArray:
        if False:
            while True:
                i = 10
        '\n        Repeat each string in the array.\n\n        If `repeats` is array-like, it is broadcast against the array and applied\n        elementwise.\n\n        Parameters\n        ----------\n        repeats : int or array-like of int\n            Number of repetitions.\n            If array-like, it is broadcast.\n\n        Returns\n        -------\n        repeated : same type as values\n            Array of repeated string objects.\n        '
        func = lambda x, y: x * y
        return self._apply(func=func, func_args=(repeats,))

    def find(self, sub: str | bytes | Any, start: int | Any=0, end: int | Any=None, side: str='left') -> T_DataArray:
        if False:
            i = 10
            return i + 15
        '\n        Return lowest or highest indexes in each strings in the array\n        where the substring is fully contained between [start:end].\n        Return -1 on failure.\n\n        If `start`, `end`, or \'sub` is array-like, they are broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        sub : str or array-like of str\n            Substring being searched.\n            If array-like, it is broadcast.\n        start : int or array-like of int\n            Left edge index.\n            If array-like, it is broadcast.\n        end : int or array-like of int\n            Right edge index.\n            If array-like, it is broadcast.\n        side : {"left", "right"}, default: "left"\n            Starting side for search.\n\n        Returns\n        -------\n        found : array of int\n        '
        sub = self._stringify(sub)
        if side == 'left':
            method = 'find'
        elif side == 'right':
            method = 'rfind'
        else:
            raise ValueError('Invalid side')
        func = lambda x, isub, istart, iend: getattr(x, method)(isub, istart, iend)
        return self._apply(func=func, func_args=(sub, start, end), dtype=int)

    def rfind(self, sub: str | bytes | Any, start: int | Any=0, end: int | Any=None) -> T_DataArray:
        if False:
            for i in range(10):
                print('nop')
        "\n        Return highest indexes in each strings in the array\n        where the substring is fully contained between [start:end].\n        Return -1 on failure.\n\n        If `start`, `end`, or 'sub` is array-like, they are broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        sub : str or array-like of str\n            Substring being searched.\n            If array-like, it is broadcast.\n        start : int or array-like of int\n            Left edge index.\n            If array-like, it is broadcast.\n        end : int or array-like of int\n            Right edge index.\n            If array-like, it is broadcast.\n\n        Returns\n        -------\n        found : array of int\n        "
        return self.find(sub, start=start, end=end, side='right')

    def index(self, sub: str | bytes | Any, start: int | Any=0, end: int | Any=None, side: str='left') -> T_DataArray:
        if False:
            while True:
                i = 10
        '\n        Return lowest or highest indexes in each strings where the substring is\n        fully contained between [start:end]. This is the same as\n        ``str.find`` except instead of returning -1, it raises a ValueError\n        when the substring is not found.\n\n        If `start`, `end`, or \'sub` is array-like, they are broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        sub : str or array-like of str\n            Substring being searched.\n            If array-like, it is broadcast.\n        start : int or array-like of int\n            Left edge index.\n            If array-like, it is broadcast.\n        end : int or array-like of int\n            Right edge index.\n            If array-like, it is broadcast.\n        side : {"left", "right"}, default: "left"\n            Starting side for search.\n\n        Returns\n        -------\n        found : array of int\n\n        Raises\n        ------\n        ValueError\n            substring is not found\n        '
        sub = self._stringify(sub)
        if side == 'left':
            method = 'index'
        elif side == 'right':
            method = 'rindex'
        else:
            raise ValueError('Invalid side')
        func = lambda x, isub, istart, iend: getattr(x, method)(isub, istart, iend)
        return self._apply(func=func, func_args=(sub, start, end), dtype=int)

    def rindex(self, sub: str | bytes | Any, start: int | Any=0, end: int | Any=None) -> T_DataArray:
        if False:
            return 10
        "\n        Return highest indexes in each strings where the substring is\n        fully contained between [start:end]. This is the same as\n        ``str.rfind`` except instead of returning -1, it raises a ValueError\n        when the substring is not found.\n\n        If `start`, `end`, or 'sub` is array-like, they are broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        sub : str or array-like of str\n            Substring being searched.\n            If array-like, it is broadcast.\n        start : int or array-like of int\n            Left edge index.\n            If array-like, it is broadcast.\n        end : int or array-like of int\n            Right edge index.\n            If array-like, it is broadcast.\n\n        Returns\n        -------\n        found : array of int\n\n        Raises\n        ------\n        ValueError\n            substring is not found\n        "
        return self.index(sub, start=start, end=end, side='right')

    def replace(self, pat: str | bytes | Pattern | Any, repl: str | bytes | Callable | Any, n: int | Any=-1, case: bool | None=None, flags: int=0, regex: bool=True) -> T_DataArray:
        if False:
            i = 10
            return i + 15
        "\n        Replace occurrences of pattern/regex in the array with some string.\n\n        If `pat`, `repl`, or 'n` is array-like, they are broadcast\n        against the array and applied elementwise.\n\n        Parameters\n        ----------\n        pat : str or re.Pattern or array-like of str or re.Pattern\n            String can be a character sequence or regular expression.\n            If array-like, it is broadcast.\n        repl : str or callable or array-like of str or callable\n            Replacement string or a callable. The callable is passed the regex\n            match object and must return a replacement string to be used.\n            See :func:`re.sub`.\n            If array-like, it is broadcast.\n        n : int or array of int, default: -1\n            Number of replacements to make from start. Use ``-1`` to replace all.\n            If array-like, it is broadcast.\n        case : bool, default: True\n            If True, case sensitive.\n            Cannot be set if `pat` is a compiled regex.\n            Equivalent to setting the `re.IGNORECASE` flag.\n        flags : int, default: 0\n            Flags to pass through to the re module, e.g. `re.IGNORECASE`.\n            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.\n            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.\n            Cannot be set if `pat` is a compiled regex.\n        regex : bool, default: True\n            If True, assumes the passed-in pattern is a regular expression.\n            If False, treats the pattern as a literal string.\n            Cannot be set to False if `pat` is a compiled regex or `repl` is\n            a callable.\n\n        Returns\n        -------\n        replaced : same type as values\n            A copy of the object with all matching occurrences of `pat`\n            replaced by `repl`.\n        "
        if _contains_str_like(repl):
            repl = self._stringify(repl)
        elif not _contains_callable(repl):
            raise TypeError('repl must be a string or callable')
        is_compiled_re = _contains_compiled_re(pat)
        if not regex and is_compiled_re:
            raise ValueError('Cannot use a compiled regex as replacement pattern with regex=False')
        if not regex and callable(repl):
            raise ValueError('Cannot use a callable replacement when regex=False')
        if regex:
            pat = self._re_compile(pat=pat, flags=flags, case=case)
            func = lambda x, ipat, irepl, i_n: ipat.sub(repl=irepl, string=x, count=i_n if i_n >= 0 else 0)
        else:
            pat = self._stringify(pat)
            func = lambda x, ipat, irepl, i_n: x.replace(ipat, irepl, i_n)
        return self._apply(func=func, func_args=(pat, repl, n))

    def extract(self, pat: str | bytes | Pattern | Any, dim: Hashable, case: bool | None=None, flags: int=0) -> T_DataArray:
        if False:
            while True:
                i = 10
        '\n        Extract the first match of capture groups in the regex pat as a new\n        dimension in a DataArray.\n\n        For each string in the DataArray, extract groups from the first match\n        of regular expression pat.\n\n        If `pat` is array-like, it is broadcast against the array and applied\n        elementwise.\n\n        Parameters\n        ----------\n        pat : str or re.Pattern or array-like of str or re.Pattern\n            A string containing a regular expression or a compiled regular\n            expression object. If array-like, it is broadcast.\n        dim : hashable or None\n            Name of the new dimension to store the captured strings in.\n            If None, the pattern must have only one capture group and the\n            resulting DataArray will have the same size as the original.\n        case : bool, default: True\n            If True, case sensitive.\n            Cannot be set if `pat` is a compiled regex.\n            Equivalent to setting the `re.IGNORECASE` flag.\n        flags : int, default: 0\n            Flags to pass through to the re module, e.g. `re.IGNORECASE`.\n            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.\n            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.\n            Cannot be set if `pat` is a compiled regex.\n\n        Returns\n        -------\n        extracted : same type as values or object array\n\n        Raises\n        ------\n        ValueError\n            `pat` has no capture groups.\n        ValueError\n            `dim` is None and there is more than one capture group.\n        ValueError\n            `case` is set when `pat` is a compiled regular expression.\n        KeyError\n            The given dimension is already present in the DataArray.\n\n        Examples\n        --------\n        Create a string array\n\n        >>> value = xr.DataArray(\n        ...     [\n        ...         [\n        ...             "a_Xy_0",\n        ...             "ab_xY_10-bab_Xy_110-baab_Xy_1100",\n        ...             "abc_Xy_01-cbc_Xy_2210",\n        ...         ],\n        ...         [\n        ...             "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",\n        ...             "",\n        ...             "abcdef_Xy_101-fef_Xy_5543210",\n        ...         ],\n        ...     ],\n        ...     dims=["X", "Y"],\n        ... )\n\n        Extract matches\n\n        >>> value.str.extract(r"(\\w+)_Xy_(\\d*)", dim="match")\n        <xarray.DataArray (X: 2, Y: 3, match: 2)>\n        array([[[\'a\', \'0\'],\n                [\'bab\', \'110\'],\n                [\'abc\', \'01\']],\n        <BLANKLINE>\n               [[\'abcd\', \'\'],\n                [\'\', \'\'],\n                [\'abcdef\', \'101\']]], dtype=\'<U6\')\n        Dimensions without coordinates: X, Y, match\n\n        See Also\n        --------\n        DataArray.str.extractall\n        DataArray.str.findall\n        re.compile\n        re.search\n        pandas.Series.str.extract\n        '
        pat = self._re_compile(pat=pat, flags=flags, case=case)
        if isinstance(pat, re.Pattern):
            maxgroups = pat.groups
        else:
            maxgroups = _apply_str_ufunc(obj=pat, func=lambda x: x.groups, dtype=np.int_).max().data.tolist()
        if maxgroups == 0:
            raise ValueError('No capture groups found in pattern.')
        if dim is None and maxgroups != 1:
            raise ValueError('Dimension must be specified if more than one capture group is given.')
        if dim is not None and dim in self._obj.dims:
            raise KeyError(f"Dimension '{dim}' already present in DataArray.")

        def _get_res_single(val, pat):
            if False:
                print('Hello World!')
            match = pat.search(val)
            if match is None:
                return ''
            res = match.group(1)
            if res is None:
                res = ''
            return res

        def _get_res_multi(val, pat):
            if False:
                print('Hello World!')
            match = pat.search(val)
            if match is None:
                return np.array([''], val.dtype)
            match = match.groups()
            match = [grp if grp is not None else '' for grp in match]
            return np.array(match, val.dtype)
        if dim is None:
            return self._apply(func=_get_res_single, func_args=(pat,))
        else:
            return duck_array_ops.astype(self._apply(func=_get_res_multi, func_args=(pat,), dtype=np.object_, output_core_dims=[[dim]], output_sizes={dim: maxgroups}), self._obj.dtype.kind)

    def extractall(self, pat: str | bytes | Pattern | Any, group_dim: Hashable, match_dim: Hashable, case: bool | None=None, flags: int=0) -> T_DataArray:
        if False:
            print('Hello World!')
        '\n        Extract all matches of capture groups in the regex pat as new\n        dimensions in a DataArray.\n\n        For each string in the DataArray, extract groups from all matches\n        of regular expression pat.\n        Equivalent to applying re.findall() to all the elements in the DataArray\n        and splitting the results across dimensions.\n\n        If `pat` is array-like, it is broadcast against the array and applied\n        elementwise.\n\n        Parameters\n        ----------\n        pat : str or re.Pattern\n            A string containing a regular expression or a compiled regular\n            expression object. If array-like, it is broadcast.\n        group_dim : hashable\n            Name of the new dimensions corresponding to the capture groups.\n            This dimension is added to the new DataArray first.\n        match_dim : hashable\n            Name of the new dimensions corresponding to the matches for each group.\n            This dimension is added to the new DataArray second.\n        case : bool, default: True\n            If True, case sensitive.\n            Cannot be set if `pat` is a compiled regex.\n            Equivalent to setting the `re.IGNORECASE` flag.\n        flags : int, default: 0\n            Flags to pass through to the re module, e.g. `re.IGNORECASE`.\n            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.\n            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.\n            Cannot be set if `pat` is a compiled regex.\n\n        Returns\n        -------\n        extracted : same type as values or object array\n\n        Raises\n        ------\n        ValueError\n            `pat` has no capture groups.\n        ValueError\n            `case` is set when `pat` is a compiled regular expression.\n        KeyError\n            Either of the given dimensions is already present in the DataArray.\n        KeyError\n            The given dimensions names are the same.\n\n        Examples\n        --------\n        Create a string array\n\n        >>> value = xr.DataArray(\n        ...     [\n        ...         [\n        ...             "a_Xy_0",\n        ...             "ab_xY_10-bab_Xy_110-baab_Xy_1100",\n        ...             "abc_Xy_01-cbc_Xy_2210",\n        ...         ],\n        ...         [\n        ...             "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",\n        ...             "",\n        ...             "abcdef_Xy_101-fef_Xy_5543210",\n        ...         ],\n        ...     ],\n        ...     dims=["X", "Y"],\n        ... )\n\n        Extract matches\n\n        >>> value.str.extractall(\n        ...     r"(\\w+)_Xy_(\\d*)", group_dim="group", match_dim="match"\n        ... )\n        <xarray.DataArray (X: 2, Y: 3, group: 3, match: 2)>\n        array([[[[\'a\', \'0\'],\n                 [\'\', \'\'],\n                 [\'\', \'\']],\n        <BLANKLINE>\n                [[\'bab\', \'110\'],\n                 [\'baab\', \'1100\'],\n                 [\'\', \'\']],\n        <BLANKLINE>\n                [[\'abc\', \'01\'],\n                 [\'cbc\', \'2210\'],\n                 [\'\', \'\']]],\n        <BLANKLINE>\n        <BLANKLINE>\n               [[[\'abcd\', \'\'],\n                 [\'dcd\', \'33210\'],\n                 [\'dccd\', \'332210\']],\n        <BLANKLINE>\n                [[\'\', \'\'],\n                 [\'\', \'\'],\n                 [\'\', \'\']],\n        <BLANKLINE>\n                [[\'abcdef\', \'101\'],\n                 [\'fef\', \'5543210\'],\n                 [\'\', \'\']]]], dtype=\'<U7\')\n        Dimensions without coordinates: X, Y, group, match\n\n        See Also\n        --------\n        DataArray.str.extract\n        DataArray.str.findall\n        re.compile\n        re.findall\n        pandas.Series.str.extractall\n        '
        pat = self._re_compile(pat=pat, flags=flags, case=case)
        if group_dim in self._obj.dims:
            raise KeyError(f"Group dimension '{group_dim}' already present in DataArray.")
        if match_dim in self._obj.dims:
            raise KeyError(f"Match dimension '{match_dim}' already present in DataArray.")
        if group_dim == match_dim:
            raise KeyError(f"Group dimension '{group_dim}' is the same as match dimension '{match_dim}'.")
        _get_count = lambda x, ipat: len(ipat.findall(x))
        maxcount = self._apply(func=_get_count, func_args=(pat,), dtype=np.int_).max().data.tolist()
        if isinstance(pat, re.Pattern):
            maxgroups = pat.groups
        else:
            maxgroups = _apply_str_ufunc(obj=pat, func=lambda x: x.groups, dtype=np.int_).max().data.tolist()

        def _get_res(val, ipat, imaxcount=maxcount, dtype=self._obj.dtype):
            if False:
                while True:
                    i = 10
            if ipat.groups == 0:
                raise ValueError('No capture groups found in pattern.')
            matches = ipat.findall(val)
            res = np.zeros([maxcount, ipat.groups], dtype)
            if ipat.groups == 1:
                for (imatch, match) in enumerate(matches):
                    res[imatch, 0] = match
            else:
                for (imatch, match) in enumerate(matches):
                    for (jmatch, submatch) in enumerate(match):
                        res[imatch, jmatch] = submatch
            return res
        return duck_array_ops.astype(self._apply(func=_get_res, func_args=(pat,), dtype=np.object_, output_core_dims=[[group_dim, match_dim]], output_sizes={group_dim: maxgroups, match_dim: maxcount}), self._obj.dtype.kind)

    def findall(self, pat: str | bytes | Pattern | Any, case: bool | None=None, flags: int=0) -> T_DataArray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Find all occurrences of pattern or regular expression in the DataArray.\n\n        Equivalent to applying re.findall() to all the elements in the DataArray.\n        Results in an object array of lists.\n        If there is only one capture group, the lists will be a sequence of matches.\n        If there are multiple capture groups, the lists will be a sequence of lists,\n        each of which contains a sequence of matches.\n\n        If `pat` is array-like, it is broadcast against the array and applied\n        elementwise.\n\n        Parameters\n        ----------\n        pat : str or re.Pattern\n            A string containing a regular expression or a compiled regular\n            expression object. If array-like, it is broadcast.\n        case : bool, default: True\n            If True, case sensitive.\n            Cannot be set if `pat` is a compiled regex.\n            Equivalent to setting the `re.IGNORECASE` flag.\n        flags : int, default: 0\n            Flags to pass through to the re module, e.g. `re.IGNORECASE`.\n            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.\n            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.\n            Cannot be set if `pat` is a compiled regex.\n\n        Returns\n        -------\n        extracted : object array\n\n        Raises\n        ------\n        ValueError\n            `pat` has no capture groups.\n        ValueError\n            `case` is set when `pat` is a compiled regular expression.\n\n        Examples\n        --------\n        Create a string array\n\n        >>> value = xr.DataArray(\n        ...     [\n        ...         [\n        ...             "a_Xy_0",\n        ...             "ab_xY_10-bab_Xy_110-baab_Xy_1100",\n        ...             "abc_Xy_01-cbc_Xy_2210",\n        ...         ],\n        ...         [\n        ...             "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",\n        ...             "",\n        ...             "abcdef_Xy_101-fef_Xy_5543210",\n        ...         ],\n        ...     ],\n        ...     dims=["X", "Y"],\n        ... )\n\n        Extract matches\n\n        >>> value.str.findall(r"(\\w+)_Xy_(\\d*)")\n        <xarray.DataArray (X: 2, Y: 3)>\n        array([[list([(\'a\', \'0\')]), list([(\'bab\', \'110\'), (\'baab\', \'1100\')]),\n                list([(\'abc\', \'01\'), (\'cbc\', \'2210\')])],\n               [list([(\'abcd\', \'\'), (\'dcd\', \'33210\'), (\'dccd\', \'332210\')]),\n                list([]), list([(\'abcdef\', \'101\'), (\'fef\', \'5543210\')])]],\n              dtype=object)\n        Dimensions without coordinates: X, Y\n\n        See Also\n        --------\n        DataArray.str.extract\n        DataArray.str.extractall\n        re.compile\n        re.findall\n        pandas.Series.str.findall\n        '
        pat = self._re_compile(pat=pat, flags=flags, case=case)

        def func(x, ipat):
            if False:
                while True:
                    i = 10
            if ipat.groups == 0:
                raise ValueError('No capture groups found in pattern.')
            return ipat.findall(x)
        return self._apply(func=func, func_args=(pat,), dtype=np.object_)

    def _partitioner(self, *, func: Callable, dim: Hashable | None, sep: str | bytes | Any | None) -> T_DataArray:
        if False:
            return 10
        '\n        Implements logic for `partition` and `rpartition`.\n        '
        sep = self._stringify(sep)
        if dim is None:
            listfunc = lambda x, isep: list(func(x, isep))
            return self._apply(func=listfunc, func_args=(sep,), dtype=np.object_)
        if not self._obj.size:
            return self._obj.copy().expand_dims({dim: 0}, axis=-1)
        arrfunc = lambda x, isep: np.array(func(x, isep), dtype=self._obj.dtype)
        return duck_array_ops.astype(self._apply(func=arrfunc, func_args=(sep,), dtype=np.object_, output_core_dims=[[dim]], output_sizes={dim: 3}), self._obj.dtype.kind)

    def partition(self, dim: Hashable | None, sep: str | bytes | Any=' ') -> T_DataArray:
        if False:
            while True:
                i = 10
        '\n        Split the strings in the DataArray at the first occurrence of separator `sep`.\n\n        This method splits the string at the first occurrence of `sep`,\n        and returns 3 elements containing the part before the separator,\n        the separator itself, and the part after the separator.\n        If the separator is not found, return 3 elements containing the string itself,\n        followed by two empty strings.\n\n        If `sep` is array-like, it is broadcast against the array and applied\n        elementwise.\n\n        Parameters\n        ----------\n        dim : hashable or None\n            Name for the dimension to place the 3 elements in.\n            If `None`, place the results as list elements in an object DataArray.\n        sep : str or bytes or array-like, default: " "\n            String to split on.\n            If array-like, it is broadcast.\n\n        Returns\n        -------\n        partitioned : same type as values or object array\n\n        See Also\n        --------\n        DataArray.str.rpartition\n        str.partition\n        pandas.Series.str.partition\n        '
        return self._partitioner(func=self._obj.dtype.type.partition, dim=dim, sep=sep)

    def rpartition(self, dim: Hashable | None, sep: str | bytes | Any=' ') -> T_DataArray:
        if False:
            i = 10
            return i + 15
        '\n        Split the strings in the DataArray at the last occurrence of separator `sep`.\n\n        This method splits the string at the last occurrence of `sep`,\n        and returns 3 elements containing the part before the separator,\n        the separator itself, and the part after the separator.\n        If the separator is not found, return 3 elements containing two empty strings,\n        followed by the string itself.\n\n        If `sep` is array-like, it is broadcast against the array and applied\n        elementwise.\n\n        Parameters\n        ----------\n        dim : hashable or None\n            Name for the dimension to place the 3 elements in.\n            If `None`, place the results as list elements in an object DataArray.\n        sep : str or bytes or array-like, default: " "\n            String to split on.\n            If array-like, it is broadcast.\n\n        Returns\n        -------\n        rpartitioned : same type as values or object array\n\n        See Also\n        --------\n        DataArray.str.partition\n        str.rpartition\n        pandas.Series.str.rpartition\n        '
        return self._partitioner(func=self._obj.dtype.type.rpartition, dim=dim, sep=sep)

    def _splitter(self, *, func: Callable, pre: bool, dim: Hashable, sep: str | bytes | Any | None, maxsplit: int) -> DataArray:
        if False:
            print('Hello World!')
        '\n        Implements logic for `split` and `rsplit`.\n        '
        if sep is not None:
            sep = self._stringify(sep)
        if dim is None:
            f_none = lambda x, isep: func(x, isep, maxsplit)
            return self._apply(func=f_none, func_args=(sep,), dtype=np.object_)
        if not self._obj.size:
            return self._obj.copy().expand_dims({dim: 0}, axis=-1)
        f_count = lambda x, isep: max(len(func(x, isep, maxsplit)), 1)
        maxsplit = self._apply(func=f_count, func_args=(sep,), dtype=np.int_).max().data.item() - 1

        def _dosplit(mystr, sep, maxsplit=maxsplit, dtype=self._obj.dtype):
            if False:
                print('Hello World!')
            res = func(mystr, sep, maxsplit)
            if len(res) < maxsplit + 1:
                pad = [''] * (maxsplit + 1 - len(res))
                if pre:
                    res += pad
                else:
                    res = pad + res
            return np.array(res, dtype=dtype)
        return duck_array_ops.astype(self._apply(func=_dosplit, func_args=(sep,), dtype=np.object_, output_core_dims=[[dim]], output_sizes={dim: maxsplit}), self._obj.dtype.kind)

    def split(self, dim: Hashable | None, sep: str | bytes | Any=None, maxsplit: int=-1) -> DataArray:
        if False:
            return 10
        '\n        Split strings in a DataArray around the given separator/delimiter `sep`.\n\n        Splits the string in the DataArray from the beginning,\n        at the specified delimiter string.\n\n        If `sep` is array-like, it is broadcast against the array and applied\n        elementwise.\n\n        Parameters\n        ----------\n        dim : hashable or None\n            Name for the dimension to place the results in.\n            If `None`, place the results as list elements in an object DataArray.\n        sep : str, default: None\n            String to split on. If ``None`` (the default), split on any whitespace.\n            If array-like, it is broadcast.\n        maxsplit : int, default: -1\n            Limit number of splits in output, starting from the beginning.\n            If -1 (the default), return all splits.\n\n        Returns\n        -------\n        splitted : same type as values or object array\n\n        Examples\n        --------\n        Create a string DataArray\n\n        >>> values = xr.DataArray(\n        ...     [\n        ...         ["abc def", "spam\\t\\teggs\\tswallow", "red_blue"],\n        ...         ["test0\\ntest1\\ntest2\\n\\ntest3", "", "abra  ka\\nda\\tbra"],\n        ...     ],\n        ...     dims=["X", "Y"],\n        ... )\n\n        Split once and put the results in a new dimension\n\n        >>> values.str.split(dim="splitted", maxsplit=1)\n        <xarray.DataArray (X: 2, Y: 3, splitted: 2)>\n        array([[[\'abc\', \'def\'],\n                [\'spam\', \'eggs\\tswallow\'],\n                [\'red_blue\', \'\']],\n        <BLANKLINE>\n               [[\'test0\', \'test1\\ntest2\\n\\ntest3\'],\n                [\'\', \'\'],\n                [\'abra\', \'ka\\nda\\tbra\']]], dtype=\'<U18\')\n        Dimensions without coordinates: X, Y, splitted\n\n        Split as many times as needed and put the results in a new dimension\n\n        >>> values.str.split(dim="splitted")\n        <xarray.DataArray (X: 2, Y: 3, splitted: 4)>\n        array([[[\'abc\', \'def\', \'\', \'\'],\n                [\'spam\', \'eggs\', \'swallow\', \'\'],\n                [\'red_blue\', \'\', \'\', \'\']],\n        <BLANKLINE>\n               [[\'test0\', \'test1\', \'test2\', \'test3\'],\n                [\'\', \'\', \'\', \'\'],\n                [\'abra\', \'ka\', \'da\', \'bra\']]], dtype=\'<U8\')\n        Dimensions without coordinates: X, Y, splitted\n\n        Split once and put the results in lists\n\n        >>> values.str.split(dim=None, maxsplit=1)\n        <xarray.DataArray (X: 2, Y: 3)>\n        array([[list([\'abc\', \'def\']), list([\'spam\', \'eggs\\tswallow\']),\n                list([\'red_blue\'])],\n               [list([\'test0\', \'test1\\ntest2\\n\\ntest3\']), list([]),\n                list([\'abra\', \'ka\\nda\\tbra\'])]], dtype=object)\n        Dimensions without coordinates: X, Y\n\n        Split as many times as needed and put the results in a list\n\n        >>> values.str.split(dim=None)\n        <xarray.DataArray (X: 2, Y: 3)>\n        array([[list([\'abc\', \'def\']), list([\'spam\', \'eggs\', \'swallow\']),\n                list([\'red_blue\'])],\n               [list([\'test0\', \'test1\', \'test2\', \'test3\']), list([]),\n                list([\'abra\', \'ka\', \'da\', \'bra\'])]], dtype=object)\n        Dimensions without coordinates: X, Y\n\n        Split only on spaces\n\n        >>> values.str.split(dim="splitted", sep=" ")\n        <xarray.DataArray (X: 2, Y: 3, splitted: 3)>\n        array([[[\'abc\', \'def\', \'\'],\n                [\'spam\\t\\teggs\\tswallow\', \'\', \'\'],\n                [\'red_blue\', \'\', \'\']],\n        <BLANKLINE>\n               [[\'test0\\ntest1\\ntest2\\n\\ntest3\', \'\', \'\'],\n                [\'\', \'\', \'\'],\n                [\'abra\', \'\', \'ka\\nda\\tbra\']]], dtype=\'<U24\')\n        Dimensions without coordinates: X, Y, splitted\n\n        See Also\n        --------\n        DataArray.str.rsplit\n        str.split\n        pandas.Series.str.split\n        '
        return self._splitter(func=self._obj.dtype.type.split, pre=True, dim=dim, sep=sep, maxsplit=maxsplit)

    def rsplit(self, dim: Hashable | None, sep: str | bytes | Any=None, maxsplit: int | Any=-1) -> DataArray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Split strings in a DataArray around the given separator/delimiter `sep`.\n\n        Splits the string in the DataArray from the end,\n        at the specified delimiter string.\n\n        If `sep` is array-like, it is broadcast against the array and applied\n        elementwise.\n\n        Parameters\n        ----------\n        dim : hashable or None\n            Name for the dimension to place the results in.\n            If `None`, place the results as list elements in an object DataArray\n        sep : str, default: None\n            String to split on. If ``None`` (the default), split on any whitespace.\n            If array-like, it is broadcast.\n        maxsplit : int, default: -1\n            Limit number of splits in output, starting from the end.\n            If -1 (the default), return all splits.\n            The final number of split values may be less than this if there are no\n            DataArray elements with that many values.\n\n        Returns\n        -------\n        rsplitted : same type as values or object array\n\n        Examples\n        --------\n        Create a string DataArray\n\n        >>> values = xr.DataArray(\n        ...     [\n        ...         ["abc def", "spam\\t\\teggs\\tswallow", "red_blue"],\n        ...         ["test0\\ntest1\\ntest2\\n\\ntest3", "", "abra  ka\\nda\\tbra"],\n        ...     ],\n        ...     dims=["X", "Y"],\n        ... )\n\n        Split once and put the results in a new dimension\n\n        >>> values.str.rsplit(dim="splitted", maxsplit=1)\n        <xarray.DataArray (X: 2, Y: 3, splitted: 2)>\n        array([[[\'abc\', \'def\'],\n                [\'spam\\t\\teggs\', \'swallow\'],\n                [\'\', \'red_blue\']],\n        <BLANKLINE>\n               [[\'test0\\ntest1\\ntest2\', \'test3\'],\n                [\'\', \'\'],\n                [\'abra  ka\\nda\', \'bra\']]], dtype=\'<U17\')\n        Dimensions without coordinates: X, Y, splitted\n\n        Split as many times as needed and put the results in a new dimension\n\n        >>> values.str.rsplit(dim="splitted")\n        <xarray.DataArray (X: 2, Y: 3, splitted: 4)>\n        array([[[\'\', \'\', \'abc\', \'def\'],\n                [\'\', \'spam\', \'eggs\', \'swallow\'],\n                [\'\', \'\', \'\', \'red_blue\']],\n        <BLANKLINE>\n               [[\'test0\', \'test1\', \'test2\', \'test3\'],\n                [\'\', \'\', \'\', \'\'],\n                [\'abra\', \'ka\', \'da\', \'bra\']]], dtype=\'<U8\')\n        Dimensions without coordinates: X, Y, splitted\n\n        Split once and put the results in lists\n\n        >>> values.str.rsplit(dim=None, maxsplit=1)\n        <xarray.DataArray (X: 2, Y: 3)>\n        array([[list([\'abc\', \'def\']), list([\'spam\\t\\teggs\', \'swallow\']),\n                list([\'red_blue\'])],\n               [list([\'test0\\ntest1\\ntest2\', \'test3\']), list([]),\n                list([\'abra  ka\\nda\', \'bra\'])]], dtype=object)\n        Dimensions without coordinates: X, Y\n\n        Split as many times as needed and put the results in a list\n\n        >>> values.str.rsplit(dim=None)\n        <xarray.DataArray (X: 2, Y: 3)>\n        array([[list([\'abc\', \'def\']), list([\'spam\', \'eggs\', \'swallow\']),\n                list([\'red_blue\'])],\n               [list([\'test0\', \'test1\', \'test2\', \'test3\']), list([]),\n                list([\'abra\', \'ka\', \'da\', \'bra\'])]], dtype=object)\n        Dimensions without coordinates: X, Y\n\n        Split only on spaces\n\n        >>> values.str.rsplit(dim="splitted", sep=" ")\n        <xarray.DataArray (X: 2, Y: 3, splitted: 3)>\n        array([[[\'\', \'abc\', \'def\'],\n                [\'\', \'\', \'spam\\t\\teggs\\tswallow\'],\n                [\'\', \'\', \'red_blue\']],\n        <BLANKLINE>\n               [[\'\', \'\', \'test0\\ntest1\\ntest2\\n\\ntest3\'],\n                [\'\', \'\', \'\'],\n                [\'abra\', \'\', \'ka\\nda\\tbra\']]], dtype=\'<U24\')\n        Dimensions without coordinates: X, Y, splitted\n\n        See Also\n        --------\n        DataArray.str.split\n        str.rsplit\n        pandas.Series.str.rsplit\n        '
        return self._splitter(func=self._obj.dtype.type.rsplit, pre=False, dim=dim, sep=sep, maxsplit=maxsplit)

    def get_dummies(self, dim: Hashable, sep: str | bytes | Any='|') -> DataArray:
        if False:
            while True:
                i = 10
        '\n        Return DataArray of dummy/indicator variables.\n\n        Each string in the DataArray is split at `sep`.\n        A new dimension is created with coordinates for each unique result,\n        and the corresponding element of that dimension is `True` if\n        that result is present and `False` if not.\n\n        If `sep` is array-like, it is broadcast against the array and applied\n        elementwise.\n\n        Parameters\n        ----------\n        dim : hashable\n            Name for the dimension to place the results in.\n        sep : str, default: "|".\n            String to split on.\n            If array-like, it is broadcast.\n\n        Returns\n        -------\n        dummies : array of bool\n\n        Examples\n        --------\n        Create a string array\n\n        >>> values = xr.DataArray(\n        ...     [\n        ...         ["a|ab~abc|abc", "ab", "a||abc|abcd"],\n        ...         ["abcd|ab|a", "abc|ab~abc", "|a"],\n        ...     ],\n        ...     dims=["X", "Y"],\n        ... )\n\n        Extract dummy values\n\n        >>> values.str.get_dummies(dim="dummies")\n        <xarray.DataArray (X: 2, Y: 3, dummies: 5)>\n        array([[[ True, False,  True, False,  True],\n                [False,  True, False, False, False],\n                [ True, False,  True,  True, False]],\n        <BLANKLINE>\n               [[ True,  True, False,  True, False],\n                [False, False,  True, False,  True],\n                [ True, False, False, False, False]]])\n        Coordinates:\n          * dummies  (dummies) <U6 \'a\' \'ab\' \'abc\' \'abcd\' \'ab~abc\'\n        Dimensions without coordinates: X, Y\n\n        See Also\n        --------\n        pandas.Series.str.get_dummies\n        '
        if not self._obj.size:
            return self._obj.copy().expand_dims({dim: 0}, axis=-1)
        sep = self._stringify(sep)
        f_set = lambda x, isep: set(x.split(isep)) - {self._stringify('')}
        setarr = self._apply(func=f_set, func_args=(sep,), dtype=np.object_)
        vals = sorted(reduce(set_union, setarr.data.ravel()))
        func = lambda x: np.array([val in x for val in vals], dtype=np.bool_)
        res = _apply_str_ufunc(func=func, obj=setarr, output_core_dims=[[dim]], output_sizes={dim: len(vals)}, dtype=np.bool_)
        res.coords[dim] = vals
        return res

    def decode(self, encoding: str, errors: str='strict') -> T_DataArray:
        if False:
            i = 10
            return i + 15
        '\n        Decode character string in the array using indicated encoding.\n\n        Parameters\n        ----------\n        encoding : str\n            The encoding to use.\n            Please see the Python documentation `codecs standard encoders <https://docs.python.org/3/library/codecs.html#standard-encodings>`_\n            section for a list of encodings handlers.\n        errors : str, default: "strict"\n            The handler for encoding errors.\n            Please see the Python documentation `codecs error handlers <https://docs.python.org/3/library/codecs.html#error-handlers>`_\n            for a list of error handlers.\n\n        Returns\n        -------\n        decoded : same type as values\n        '
        if encoding in _cpython_optimized_decoders:
            func = lambda x: x.decode(encoding, errors)
        else:
            decoder = codecs.getdecoder(encoding)
            func = lambda x: decoder(x, errors)[0]
        return self._apply(func=func, dtype=np.str_)

    def encode(self, encoding: str, errors: str='strict') -> T_DataArray:
        if False:
            return 10
        '\n        Encode character string in the array using indicated encoding.\n\n        Parameters\n        ----------\n        encoding : str\n            The encoding to use.\n            Please see the Python documentation `codecs standard encoders <https://docs.python.org/3/library/codecs.html#standard-encodings>`_\n            section for a list of encodings handlers.\n        errors : str, default: "strict"\n            The handler for encoding errors.\n            Please see the Python documentation `codecs error handlers <https://docs.python.org/3/library/codecs.html#error-handlers>`_\n            for a list of error handlers.\n\n        Returns\n        -------\n        encoded : same type as values\n        '
        if encoding in _cpython_optimized_encoders:
            func = lambda x: x.encode(encoding, errors)
        else:
            encoder = codecs.getencoder(encoding)
            func = lambda x: encoder(x, errors)[0]
        return self._apply(func=func, dtype=np.bytes_)