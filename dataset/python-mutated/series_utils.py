"""
Implement Series's accessors public API as pandas does.

Accessors: `Series.cat`, `Series.str`, `Series.dt`
"""
import re
from typing import TYPE_CHECKING
import numpy as np
import pandas
from modin.logging import ClassLogger
from modin.utils import _inherit_docstrings
if TYPE_CHECKING:
    from datetime import tzinfo
    from pandas._typing import npt

@_inherit_docstrings(pandas.core.arrays.categorical.CategoricalAccessor)
class CategoryMethods(ClassLogger):

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self._series = data
        self._query_compiler = data._query_compiler

    @pandas.util.cache_readonly
    def _Series(self):
        if False:
            i = 10
            return i + 15
        from .series import Series
        return Series

    @property
    def categories(self):
        if False:
            i = 10
            return i + 15
        return self._series.dtype.categories

    @categories.setter
    def categories(self, categories):
        if False:
            i = 10
            return i + 15

        def set_categories(series, categories):
            if False:
                while True:
                    i = 10
            series.cat.categories = categories
        self._series._default_to_pandas(set_categories, categories=categories)

    @property
    def ordered(self):
        if False:
            print('Hello World!')
        return self._series.dtype.ordered

    @property
    def codes(self):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.cat_codes())

    def rename_categories(self, new_categories):
        if False:
            i = 10
            return i + 15
        return self._default_to_pandas(pandas.Series.cat.rename_categories, new_categories)

    def reorder_categories(self, new_categories, ordered=None):
        if False:
            i = 10
            return i + 15
        return self._default_to_pandas(pandas.Series.cat.reorder_categories, new_categories, ordered=ordered)

    def add_categories(self, new_categories):
        if False:
            i = 10
            return i + 15
        return self._default_to_pandas(pandas.Series.cat.add_categories, new_categories)

    def remove_categories(self, removals):
        if False:
            return 10
        return self._default_to_pandas(pandas.Series.cat.remove_categories, removals)

    def remove_unused_categories(self):
        if False:
            print('Hello World!')
        return self._default_to_pandas(pandas.Series.cat.remove_unused_categories)

    def set_categories(self, new_categories, ordered=None, rename=False):
        if False:
            for i in range(10):
                print('nop')
        return self._default_to_pandas(pandas.Series.cat.set_categories, new_categories, ordered=ordered, rename=rename)

    def as_ordered(self):
        if False:
            i = 10
            return i + 15
        return self._default_to_pandas(pandas.Series.cat.as_ordered)

    def as_unordered(self):
        if False:
            for i in range(10):
                print('nop')
        return self._default_to_pandas(pandas.Series.cat.as_unordered)

    def _default_to_pandas(self, op, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Convert `self` to pandas type and call a pandas cat.`op` on it.\n\n        Parameters\n        ----------\n        op : str\n            Name of pandas function.\n        *args : list\n            Additional positional arguments to be passed in `op`.\n        **kwargs : dict\n            Additional keywords arguments to be passed in `op`.\n\n        Returns\n        -------\n        object\n            Result of operation.\n        '
        return self._series._default_to_pandas(lambda series: op(series.cat, *args, **kwargs))

@_inherit_docstrings(pandas.core.strings.accessor.StringMethods)
class StringMethods(ClassLogger):

    def __init__(self, data):
        if False:
            print('Hello World!')
        self._series = data
        self._query_compiler = data._query_compiler

    @pandas.util.cache_readonly
    def _Series(self):
        if False:
            return 10
        from .series import Series
        return Series

    def casefold(self):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.str_casefold())

    def cat(self, others=None, sep=None, na_rep=None, join='left'):
        if False:
            i = 10
            return i + 15
        if isinstance(others, self._Series):
            others = others._to_pandas()
        compiler_result = self._query_compiler.str_cat(others=others, sep=sep, na_rep=na_rep, join=join)
        return compiler_result.to_pandas().squeeze() if others is None else self._Series(query_compiler=compiler_result)

    def decode(self, encoding, errors='strict'):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.str_decode(encoding, errors))

    def split(self, pat=None, *, n=-1, expand=False, regex=None):
        if False:
            print('Hello World!')
        if expand:
            from .dataframe import DataFrame
            return DataFrame(query_compiler=self._query_compiler.str_split(pat=pat, n=n, expand=True, regex=regex))
        else:
            return self._Series(query_compiler=self._query_compiler.str_split(pat=pat, n=n, expand=expand, regex=regex))

    def rsplit(self, pat=None, *, n=-1, expand=False):
        if False:
            return 10
        if not pat and pat is not None:
            raise ValueError('rsplit() requires a non-empty pattern match.')
        if expand:
            from .dataframe import DataFrame
            return DataFrame(query_compiler=self._query_compiler.str_rsplit(pat=pat, n=n, expand=True))
        else:
            return self._Series(query_compiler=self._query_compiler.str_rsplit(pat=pat, n=n, expand=expand))

    def get(self, i):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.str_get(i))

    def join(self, sep):
        if False:
            print('Hello World!')
        if sep is None:
            raise AttributeError("'NoneType' object has no attribute 'join'")
        return self._Series(query_compiler=self._query_compiler.str_join(sep))

    def get_dummies(self, sep='|'):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.str_get_dummies(sep))

    def contains(self, pat, case=True, flags=0, na=None, regex=True):
        if False:
            i = 10
            return i + 15
        if pat is None and (not case):
            raise AttributeError("'NoneType' object has no attribute 'upper'")
        return self._Series(query_compiler=self._query_compiler.str_contains(pat, case=case, flags=flags, na=na, regex=regex))

    def replace(self, pat, repl, n=-1, case=None, flags=0, regex=False):
        if False:
            return 10
        if not (isinstance(repl, str) or callable(repl)):
            raise TypeError('repl must be a string or callable')
        return self._Series(query_compiler=self._query_compiler.str_replace(pat, repl, n=n, case=case, flags=flags, regex=regex))

    def pad(self, width, side='left', fillchar=' '):
        if False:
            for i in range(10):
                print('nop')
        if len(fillchar) != 1:
            raise TypeError('fillchar must be a character, not str')
        return self._Series(query_compiler=self._query_compiler.str_pad(width, side=side, fillchar=fillchar))

    def center(self, width, fillchar=' '):
        if False:
            i = 10
            return i + 15
        if len(fillchar) != 1:
            raise TypeError('fillchar must be a character, not str')
        return self._Series(query_compiler=self._query_compiler.str_center(width, fillchar=fillchar))

    def ljust(self, width, fillchar=' '):
        if False:
            i = 10
            return i + 15
        if len(fillchar) != 1:
            raise TypeError('fillchar must be a character, not str')
        return self._Series(query_compiler=self._query_compiler.str_ljust(width, fillchar=fillchar))

    def rjust(self, width, fillchar=' '):
        if False:
            return 10
        if len(fillchar) != 1:
            raise TypeError('fillchar must be a character, not str')
        return self._Series(query_compiler=self._query_compiler.str_rjust(width, fillchar=fillchar))

    def zfill(self, width):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.str_zfill(width))

    def wrap(self, width, **kwargs):
        if False:
            print('Hello World!')
        if width <= 0:
            raise ValueError('invalid width {} (must be > 0)'.format(width))
        return self._Series(query_compiler=self._query_compiler.str_wrap(width, **kwargs))

    def slice(self, start=None, stop=None, step=None):
        if False:
            while True:
                i = 10
        if step == 0:
            raise ValueError('slice step cannot be zero')
        return self._Series(query_compiler=self._query_compiler.str_slice(start=start, stop=stop, step=step))

    def slice_replace(self, start=None, stop=None, repl=None):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.str_slice_replace(start=start, stop=stop, repl=repl))

    def count(self, pat, flags=0):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(pat, (str, re.Pattern)):
            raise TypeError('first argument must be string or compiled pattern')
        return self._Series(query_compiler=self._query_compiler.str_count(pat, flags=flags))

    def startswith(self, pat, na=None):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.str_startswith(pat, na=na))

    def encode(self, encoding, errors='strict'):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.str_encode(encoding, errors))

    def endswith(self, pat, na=None):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.str_endswith(pat, na=na))

    def findall(self, pat, flags=0):
        if False:
            while True:
                i = 10
        if not isinstance(pat, (str, re.Pattern)):
            raise TypeError('first argument must be string or compiled pattern')
        return self._Series(query_compiler=self._query_compiler.str_findall(pat, flags=flags))

    def fullmatch(self, pat, case=True, flags=0, na=None):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(pat, (str, re.Pattern)):
            raise TypeError('first argument must be string or compiled pattern')
        return self._Series(query_compiler=self._query_compiler.str_fullmatch(pat, case=case, flags=flags, na=na))

    def match(self, pat, case=True, flags=0, na=None):
        if False:
            i = 10
            return i + 15
        if not isinstance(pat, (str, re.Pattern)):
            raise TypeError('first argument must be string or compiled pattern')
        return self._Series(query_compiler=self._query_compiler.str_match(pat, case=case, flags=flags, na=na))

    def extract(self, pat, flags=0, expand=True):
        if False:
            for i in range(10):
                print('nop')
        query_compiler = self._query_compiler.str_extract(pat, flags=flags, expand=expand)
        from .dataframe import DataFrame
        return DataFrame(query_compiler=query_compiler) if expand or re.compile(pat).groups > 1 else self._Series(query_compiler=query_compiler)

    def extractall(self, pat, flags=0):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.str_extractall(pat, flags))

    def len(self):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.str_len())

    def strip(self, to_strip=None):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.str_strip(to_strip=to_strip))

    def rstrip(self, to_strip=None):
        if False:
            while True:
                i = 10
        return self._Series(query_compiler=self._query_compiler.str_rstrip(to_strip=to_strip))

    def lstrip(self, to_strip=None):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.str_lstrip(to_strip=to_strip))

    def partition(self, sep=' ', expand=True):
        if False:
            for i in range(10):
                print('nop')
        if sep is not None and len(sep) == 0:
            raise ValueError('empty separator')
        from .dataframe import DataFrame
        return (DataFrame if expand else self._Series)(query_compiler=self._query_compiler.str_partition(sep=sep, expand=expand))

    def removeprefix(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.str_removeprefix(prefix))

    def removesuffix(self, suffix):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.str_removesuffix(suffix))

    def repeat(self, repeats):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.str_repeat(repeats))

    def rpartition(self, sep=' ', expand=True):
        if False:
            return 10
        if sep is not None and len(sep) == 0:
            raise ValueError('empty separator')
        from .dataframe import DataFrame
        return (DataFrame if expand else self._Series)(query_compiler=self._query_compiler.str_rpartition(sep=sep, expand=expand))

    def lower(self):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.str_lower())

    def upper(self):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.str_upper())

    def title(self):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.str_title())

    def find(self, sub, start=0, end=None):
        if False:
            i = 10
            return i + 15
        if not isinstance(sub, str):
            raise TypeError('expected a string object, not {0}'.format(type(sub).__name__))
        return self._Series(query_compiler=self._query_compiler.str_find(sub, start=start, end=end))

    def rfind(self, sub, start=0, end=None):
        if False:
            print('Hello World!')
        if not isinstance(sub, str):
            raise TypeError('expected a string object, not {0}'.format(type(sub).__name__))
        return self._Series(query_compiler=self._query_compiler.str_rfind(sub, start=start, end=end))

    def index(self, sub, start=0, end=None):
        if False:
            i = 10
            return i + 15
        if not isinstance(sub, str):
            raise TypeError('expected a string object, not {0}'.format(type(sub).__name__))
        return self._Series(query_compiler=self._query_compiler.str_index(sub, start=start, end=end))

    def rindex(self, sub, start=0, end=None):
        if False:
            while True:
                i = 10
        if not isinstance(sub, str):
            raise TypeError('expected a string object, not {0}'.format(type(sub).__name__))
        return self._Series(query_compiler=self._query_compiler.str_rindex(sub, start=start, end=end))

    def capitalize(self):
        if False:
            while True:
                i = 10
        return self._Series(query_compiler=self._query_compiler.str_capitalize())

    def swapcase(self):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.str_swapcase())

    def normalize(self, form):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.str_normalize(form))

    def translate(self, table):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.str_translate(table))

    def isalnum(self):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.str_isalnum())

    def isalpha(self):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.str_isalpha())

    def isdigit(self):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.str_isdigit())

    def isspace(self):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.str_isspace())

    def islower(self):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.str_islower())

    def isupper(self):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.str_isupper())

    def istitle(self):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.str_istitle())

    def isnumeric(self):
        if False:
            while True:
                i = 10
        return self._Series(query_compiler=self._query_compiler.str_isnumeric())

    def isdecimal(self):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.str_isdecimal())

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self._Series(query_compiler=self._query_compiler.str___getitem__(key))

    def _default_to_pandas(self, op, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Convert `self` to pandas type and call a pandas str.`op` on it.\n\n        Parameters\n        ----------\n        op : str\n            Name of pandas function.\n        *args : list\n            Additional positional arguments to be passed in `op`.\n        **kwargs : dict\n            Additional keywords arguments to be passed in `op`.\n\n        Returns\n        -------\n        object\n            Result of operation.\n        '
        return self._series._default_to_pandas(lambda series: op(series.str, *args, **kwargs))

@_inherit_docstrings(pandas.core.indexes.accessors.CombinedDatetimelikeProperties)
class DatetimeProperties(ClassLogger):

    def __init__(self, data):
        if False:
            return 10
        self._series = data
        self._query_compiler = data._query_compiler

    @pandas.util.cache_readonly
    def _Series(self):
        if False:
            for i in range(10):
                print('nop')
        from .series import Series
        return Series

    @property
    def date(self):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.dt_date())

    @property
    def time(self):
        if False:
            while True:
                i = 10
        return self._Series(query_compiler=self._query_compiler.dt_time())

    @property
    def timetz(self):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.dt_timetz())

    @property
    def year(self):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.dt_year())

    @property
    def month(self):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.dt_month())

    @property
    def day(self):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.dt_day())

    @property
    def hour(self):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.dt_hour())

    @property
    def minute(self):
        if False:
            while True:
                i = 10
        return self._Series(query_compiler=self._query_compiler.dt_minute())

    @property
    def second(self):
        if False:
            while True:
                i = 10
        return self._Series(query_compiler=self._query_compiler.dt_second())

    @property
    def microsecond(self):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.dt_microsecond())

    @property
    def nanosecond(self):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.dt_nanosecond())

    @property
    def dayofweek(self):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.dt_dayofweek())
    day_of_week = dayofweek

    @property
    def weekday(self):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.dt_weekday())

    @property
    def dayofyear(self):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.dt_dayofyear())
    day_of_year = dayofyear

    @property
    def quarter(self):
        if False:
            while True:
                i = 10
        return self._Series(query_compiler=self._query_compiler.dt_quarter())

    @property
    def is_month_start(self):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.dt_is_month_start())

    @property
    def is_month_end(self):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.dt_is_month_end())

    @property
    def is_quarter_start(self):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.dt_is_quarter_start())

    @property
    def is_quarter_end(self):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.dt_is_quarter_end())

    @property
    def is_year_start(self):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.dt_is_year_start())

    @property
    def is_year_end(self):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.dt_is_year_end())

    @property
    def is_leap_year(self):
        if False:
            while True:
                i = 10
        return self._Series(query_compiler=self._query_compiler.dt_is_leap_year())

    @property
    def daysinmonth(self):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.dt_daysinmonth())

    @property
    def days_in_month(self):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.dt_days_in_month())

    @property
    def tz(self) -> 'tzinfo | None':
        if False:
            while True:
                i = 10
        dtype = self._series.dtype
        if isinstance(dtype, np.dtype):
            return None
        return dtype.tz

    @property
    def freq(self):
        if False:
            return 10
        return self._query_compiler.dt_freq().to_pandas().squeeze()

    @property
    def unit(self):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.dt_unit()).iloc[0]

    def as_unit(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.dt_as_unit(*args, **kwargs))

    def to_period(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.dt_to_period(*args, **kwargs))

    def asfreq(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.dt_asfreq(*args, **kwargs))

    def to_pydatetime(self):
        if False:
            while True:
                i = 10
        return self._Series(query_compiler=self._query_compiler.dt_to_pydatetime()).to_numpy()

    def tz_localize(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.dt_tz_localize(*args, **kwargs))

    def tz_convert(self, *args, **kwargs):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.dt_tz_convert(*args, **kwargs))

    def normalize(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.dt_normalize(*args, **kwargs))

    def strftime(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self._Series(query_compiler=self._query_compiler.dt_strftime(*args, **kwargs))

    def round(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.dt_round(*args, **kwargs))

    def floor(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.dt_floor(*args, **kwargs))

    def ceil(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.dt_ceil(*args, **kwargs))

    def month_name(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.dt_month_name(*args, **kwargs))

    def day_name(self, *args, **kwargs):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.dt_day_name(*args, **kwargs))

    def total_seconds(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self._Series(query_compiler=self._query_compiler.dt_total_seconds(*args, **kwargs))

    def to_pytimedelta(self) -> 'npt.NDArray[np.object_]':
        if False:
            i = 10
            return i + 15
        res = self._query_compiler.dt_to_pytimedelta()
        return res.to_numpy()[:, 0]

    @property
    def seconds(self):
        if False:
            i = 10
            return i + 15
        return self._Series(query_compiler=self._query_compiler.dt_seconds())

    @property
    def days(self):
        if False:
            for i in range(10):
                print('nop')
        return self._Series(query_compiler=self._query_compiler.dt_days())

    @property
    def microseconds(self):
        if False:
            while True:
                i = 10
        return self._Series(query_compiler=self._query_compiler.dt_microseconds())

    @property
    def nanoseconds(self):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.dt_nanoseconds())

    @property
    def components(self):
        if False:
            i = 10
            return i + 15
        from .dataframe import DataFrame
        return DataFrame(query_compiler=self._query_compiler.dt_components())

    def isocalendar(self):
        if False:
            return 10
        from .dataframe import DataFrame
        return DataFrame(query_compiler=self._query_compiler.dt_isocalendar())

    @property
    def qyear(self):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.dt_qyear())

    @property
    def start_time(self):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.dt_start_time())

    @property
    def end_time(self):
        if False:
            return 10
        return self._Series(query_compiler=self._query_compiler.dt_end_time())

    def to_timestamp(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self._Series(query_compiler=self._query_compiler.dt_to_timestamp(*args, **kwargs))