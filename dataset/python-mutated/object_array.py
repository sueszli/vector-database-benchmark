from __future__ import annotations
import functools
import re
import textwrap
from typing import TYPE_CHECKING, Callable, Literal, cast
import unicodedata
import numpy as np
from pandas._libs import lib
import pandas._libs.missing as libmissing
import pandas._libs.ops as libops
from pandas.core.dtypes.missing import isna
from pandas.core.strings.base import BaseStringArrayMethods
if TYPE_CHECKING:
    from collections.abc import Sequence
    from pandas._typing import NpDtype, Scalar
    from pandas import Series

class ObjectStringArrayMixin(BaseStringArrayMethods):
    """
    String Methods operating on object-dtype ndarrays.
    """
    _str_na_value = np.nan

    def __len__(self) -> int:
        if False:
            return 10
        raise NotImplementedError

    def _str_map(self, f, na_value=None, dtype: NpDtype | None=None, convert: bool=True):
        if False:
            return 10
        '\n        Map a callable over valid elements of the array.\n\n        Parameters\n        ----------\n        f : Callable\n            A function to call on each non-NA element.\n        na_value : Scalar, optional\n            The value to set for NA values. Might also be used for the\n            fill value if the callable `f` raises an exception.\n            This defaults to ``self._str_na_value`` which is ``np.nan``\n            for object-dtype and Categorical and ``pd.NA`` for StringArray.\n        dtype : Dtype, optional\n            The dtype of the result array.\n        convert : bool, default True\n            Whether to call `maybe_convert_objects` on the resulting ndarray\n        '
        if dtype is None:
            dtype = np.dtype('object')
        if na_value is None:
            na_value = self._str_na_value
        if not len(self):
            return np.array([], dtype=dtype)
        arr = np.asarray(self, dtype=object)
        mask = isna(arr)
        map_convert = convert and (not np.all(mask))
        try:
            result = lib.map_infer_mask(arr, f, mask.view(np.uint8), map_convert)
        except (TypeError, AttributeError) as err:
            p_err = '((takes)|(missing)) (?(2)from \\d+ to )?\\d+ (?(3)required )positional arguments?'
            if len(err.args) >= 1 and re.search(p_err, err.args[0]):
                raise err

            def g(x):
                if False:
                    while True:
                        i = 10
                try:
                    return f(x)
                except (TypeError, AttributeError):
                    return na_value
            return self._str_map(g, na_value=na_value, dtype=dtype)
        if not isinstance(result, np.ndarray):
            return result
        if na_value is not np.nan:
            np.putmask(result, mask, na_value)
            if convert and result.dtype == object:
                result = lib.maybe_convert_objects(result)
        return result

    def _str_count(self, pat, flags: int=0):
        if False:
            print('Hello World!')
        regex = re.compile(pat, flags=flags)
        f = lambda x: len(regex.findall(x))
        return self._str_map(f, dtype='int64')

    def _str_pad(self, width: int, side: Literal['left', 'right', 'both']='left', fillchar: str=' '):
        if False:
            i = 10
            return i + 15
        if side == 'left':
            f = lambda x: x.rjust(width, fillchar)
        elif side == 'right':
            f = lambda x: x.ljust(width, fillchar)
        elif side == 'both':
            f = lambda x: x.center(width, fillchar)
        else:
            raise ValueError('Invalid side')
        return self._str_map(f)

    def _str_contains(self, pat, case: bool=True, flags: int=0, na=np.nan, regex: bool=True):
        if False:
            for i in range(10):
                print('nop')
        if regex:
            if not case:
                flags |= re.IGNORECASE
            pat = re.compile(pat, flags=flags)
            f = lambda x: pat.search(x) is not None
        elif case:
            f = lambda x: pat in x
        else:
            upper_pat = pat.upper()
            f = lambda x: upper_pat in x.upper()
        return self._str_map(f, na, dtype=np.dtype('bool'))

    def _str_startswith(self, pat, na=None):
        if False:
            for i in range(10):
                print('nop')
        f = lambda x: x.startswith(pat)
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_endswith(self, pat, na=None):
        if False:
            for i in range(10):
                print('nop')
        f = lambda x: x.endswith(pat)
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_replace(self, pat: str | re.Pattern, repl: str | Callable, n: int=-1, case: bool=True, flags: int=0, regex: bool=True):
        if False:
            return 10
        if case is False:
            flags |= re.IGNORECASE
        if regex or flags or callable(repl):
            if not isinstance(pat, re.Pattern):
                if regex is False:
                    pat = re.escape(pat)
                pat = re.compile(pat, flags=flags)
            n = n if n >= 0 else 0
            f = lambda x: pat.sub(repl=repl, string=x, count=n)
        else:
            f = lambda x: x.replace(pat, repl, n)
        return self._str_map(f, dtype=str)

    def _str_repeat(self, repeats: int | Sequence[int]):
        if False:
            while True:
                i = 10
        if lib.is_integer(repeats):
            rint = cast(int, repeats)

            def scalar_rep(x):
                if False:
                    i = 10
                    return i + 15
                try:
                    return bytes.__mul__(x, rint)
                except TypeError:
                    return str.__mul__(x, rint)
            return self._str_map(scalar_rep, dtype=str)
        else:
            from pandas.core.arrays.string_ import BaseStringArray

            def rep(x, r):
                if False:
                    print('Hello World!')
                if x is libmissing.NA:
                    return x
                try:
                    return bytes.__mul__(x, r)
                except TypeError:
                    return str.__mul__(x, r)
            result = libops.vec_binop(np.asarray(self), np.asarray(repeats, dtype=object), rep)
            if isinstance(self, BaseStringArray):
                result = type(self)._from_sequence(result)
            return result

    def _str_match(self, pat: str, case: bool=True, flags: int=0, na: Scalar | None=None):
        if False:
            return 10
        if not case:
            flags |= re.IGNORECASE
        regex = re.compile(pat, flags=flags)
        f = lambda x: regex.match(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_fullmatch(self, pat: str | re.Pattern, case: bool=True, flags: int=0, na: Scalar | None=None):
        if False:
            while True:
                i = 10
        if not case:
            flags |= re.IGNORECASE
        regex = re.compile(pat, flags=flags)
        f = lambda x: regex.fullmatch(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_encode(self, encoding, errors: str='strict'):
        if False:
            return 10
        f = lambda x: x.encode(encoding, errors=errors)
        return self._str_map(f, dtype=object)

    def _str_find(self, sub, start: int=0, end=None):
        if False:
            while True:
                i = 10
        return self._str_find_(sub, start, end, side='left')

    def _str_rfind(self, sub, start: int=0, end=None):
        if False:
            i = 10
            return i + 15
        return self._str_find_(sub, start, end, side='right')

    def _str_find_(self, sub, start, end, side):
        if False:
            for i in range(10):
                print('nop')
        if side == 'left':
            method = 'find'
        elif side == 'right':
            method = 'rfind'
        else:
            raise ValueError('Invalid side')
        if end is None:
            f = lambda x: getattr(x, method)(sub, start)
        else:
            f = lambda x: getattr(x, method)(sub, start, end)
        return self._str_map(f, dtype='int64')

    def _str_findall(self, pat, flags: int=0):
        if False:
            return 10
        regex = re.compile(pat, flags=flags)
        return self._str_map(regex.findall, dtype='object')

    def _str_get(self, i):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                while True:
                    i = 10
            if isinstance(x, dict):
                return x.get(i)
            elif len(x) > i >= -len(x):
                return x[i]
            return self._str_na_value
        return self._str_map(f)

    def _str_index(self, sub, start: int=0, end=None):
        if False:
            while True:
                i = 10
        if end:
            f = lambda x: x.index(sub, start, end)
        else:
            f = lambda x: x.index(sub, start, end)
        return self._str_map(f, dtype='int64')

    def _str_rindex(self, sub, start: int=0, end=None):
        if False:
            i = 10
            return i + 15
        if end:
            f = lambda x: x.rindex(sub, start, end)
        else:
            f = lambda x: x.rindex(sub, start, end)
        return self._str_map(f, dtype='int64')

    def _str_join(self, sep: str):
        if False:
            for i in range(10):
                print('nop')
        return self._str_map(sep.join)

    def _str_partition(self, sep: str, expand):
        if False:
            i = 10
            return i + 15
        result = self._str_map(lambda x: x.partition(sep), dtype='object')
        return result

    def _str_rpartition(self, sep: str, expand):
        if False:
            print('Hello World!')
        return self._str_map(lambda x: x.rpartition(sep), dtype='object')

    def _str_len(self):
        if False:
            return 10
        return self._str_map(len, dtype='int64')

    def _str_slice(self, start=None, stop=None, step=None):
        if False:
            return 10
        obj = slice(start, stop, step)
        return self._str_map(lambda x: x[obj])

    def _str_slice_replace(self, start=None, stop=None, repl=None):
        if False:
            while True:
                i = 10
        if repl is None:
            repl = ''

        def f(x):
            if False:
                while True:
                    i = 10
            if x[start:stop] == '':
                local_stop = start
            else:
                local_stop = stop
            y = ''
            if start is not None:
                y += x[:start]
            y += repl
            if stop is not None:
                y += x[local_stop:]
            return y
        return self._str_map(f)

    def _str_split(self, pat: str | re.Pattern | None=None, n=-1, expand: bool=False, regex: bool | None=None):
        if False:
            print('Hello World!')
        if pat is None:
            if n is None or n == 0:
                n = -1
            f = lambda x: x.split(pat, n)
        else:
            new_pat: str | re.Pattern
            if regex is True or isinstance(pat, re.Pattern):
                new_pat = re.compile(pat)
            elif regex is False:
                new_pat = pat
            elif len(pat) == 1:
                new_pat = pat
            else:
                new_pat = re.compile(pat)
            if isinstance(new_pat, re.Pattern):
                if n is None or n == -1:
                    n = 0
                f = lambda x: new_pat.split(x, maxsplit=n)
            else:
                if n is None or n == 0:
                    n = -1
                f = lambda x: x.split(pat, n)
        return self._str_map(f, dtype=object)

    def _str_rsplit(self, pat=None, n=-1):
        if False:
            print('Hello World!')
        if n is None or n == 0:
            n = -1
        f = lambda x: x.rsplit(pat, n)
        return self._str_map(f, dtype='object')

    def _str_translate(self, table):
        if False:
            for i in range(10):
                print('nop')
        return self._str_map(lambda x: x.translate(table))

    def _str_wrap(self, width: int, **kwargs):
        if False:
            while True:
                i = 10
        kwargs['width'] = width
        tw = textwrap.TextWrapper(**kwargs)
        return self._str_map(lambda s: '\n'.join(tw.wrap(s)))

    def _str_get_dummies(self, sep: str='|'):
        if False:
            while True:
                i = 10
        from pandas import Series
        arr = Series(self).fillna('')
        try:
            arr = sep + arr + sep
        except (TypeError, NotImplementedError):
            arr = sep + arr.astype(str) + sep
        tags: set[str] = set()
        for ts in Series(arr, copy=False).str.split(sep):
            tags.update(ts)
        tags2 = sorted(tags - {''})
        dummies = np.empty((len(arr), len(tags2)), dtype=np.int64)

        def _isin(test_elements: str, element: str) -> bool:
            if False:
                i = 10
                return i + 15
            return element in test_elements
        for (i, t) in enumerate(tags2):
            pat = sep + t + sep
            dummies[:, i] = lib.map_infer(arr.to_numpy(), functools.partial(_isin, element=pat))
        return (dummies, tags2)

    def _str_upper(self):
        if False:
            return 10
        return self._str_map(lambda x: x.upper())

    def _str_isalnum(self):
        if False:
            return 10
        return self._str_map(str.isalnum, dtype='bool')

    def _str_isalpha(self):
        if False:
            for i in range(10):
                print('nop')
        return self._str_map(str.isalpha, dtype='bool')

    def _str_isdecimal(self):
        if False:
            return 10
        return self._str_map(str.isdecimal, dtype='bool')

    def _str_isdigit(self):
        if False:
            while True:
                i = 10
        return self._str_map(str.isdigit, dtype='bool')

    def _str_islower(self):
        if False:
            i = 10
            return i + 15
        return self._str_map(str.islower, dtype='bool')

    def _str_isnumeric(self):
        if False:
            i = 10
            return i + 15
        return self._str_map(str.isnumeric, dtype='bool')

    def _str_isspace(self):
        if False:
            while True:
                i = 10
        return self._str_map(str.isspace, dtype='bool')

    def _str_istitle(self):
        if False:
            for i in range(10):
                print('nop')
        return self._str_map(str.istitle, dtype='bool')

    def _str_isupper(self):
        if False:
            print('Hello World!')
        return self._str_map(str.isupper, dtype='bool')

    def _str_capitalize(self):
        if False:
            return 10
        return self._str_map(str.capitalize)

    def _str_casefold(self):
        if False:
            while True:
                i = 10
        return self._str_map(str.casefold)

    def _str_title(self):
        if False:
            return 10
        return self._str_map(str.title)

    def _str_swapcase(self):
        if False:
            print('Hello World!')
        return self._str_map(str.swapcase)

    def _str_lower(self):
        if False:
            for i in range(10):
                print('nop')
        return self._str_map(str.lower)

    def _str_normalize(self, form):
        if False:
            print('Hello World!')
        f = lambda x: unicodedata.normalize(form, x)
        return self._str_map(f)

    def _str_strip(self, to_strip=None):
        if False:
            i = 10
            return i + 15
        return self._str_map(lambda x: x.strip(to_strip))

    def _str_lstrip(self, to_strip=None):
        if False:
            while True:
                i = 10
        return self._str_map(lambda x: x.lstrip(to_strip))

    def _str_rstrip(self, to_strip=None):
        if False:
            while True:
                i = 10
        return self._str_map(lambda x: x.rstrip(to_strip))

    def _str_removeprefix(self, prefix: str) -> Series:
        if False:
            for i in range(10):
                print('nop')

        def removeprefix(text: str) -> str:
            if False:
                print('Hello World!')
            if text.startswith(prefix):
                return text[len(prefix):]
            return text
        return self._str_map(removeprefix)

    def _str_removesuffix(self, suffix: str) -> Series:
        if False:
            return 10
        return self._str_map(lambda x: x.removesuffix(suffix))

    def _str_extract(self, pat: str, flags: int=0, expand: bool=True):
        if False:
            for i in range(10):
                print('nop')
        regex = re.compile(pat, flags=flags)
        na_value = self._str_na_value
        if not expand:

            def g(x):
                if False:
                    return 10
                m = regex.search(x)
                return m.groups()[0] if m else na_value
            return self._str_map(g, convert=False)
        empty_row = [na_value] * regex.groups

        def f(x):
            if False:
                return 10
            if not isinstance(x, str):
                return empty_row
            m = regex.search(x)
            if m:
                return [na_value if item is None else item for item in m.groups()]
            else:
                return empty_row
        return [f(val) for val in np.asarray(self)]