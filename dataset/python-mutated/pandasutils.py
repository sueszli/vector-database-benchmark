import functools
import shutil
import tempfile
import warnings
from contextlib import contextmanager
import decimal
from typing import Any, Union
tabulate_requirement_message = None
try:
    from tabulate import tabulate
except ImportError as e:
    tabulate_requirement_message = str(e)
have_tabulate = tabulate_requirement_message is None
matplotlib_requirement_message = None
try:
    import matplotlib
except ImportError as e:
    matplotlib_requirement_message = str(e)
have_matplotlib = matplotlib_requirement_message is None
plotly_requirement_message = None
try:
    import plotly
except ImportError as e:
    plotly_requirement_message = str(e)
have_plotly = plotly_requirement_message is None
try:
    from pyspark.sql.pandas.utils import require_minimum_pandas_version
    require_minimum_pandas_version()
    import pandas as pd
except ImportError:
    pass
import pyspark.pandas as ps
from pyspark.pandas.frame import DataFrame
from pyspark.pandas.indexes import Index
from pyspark.pandas.series import Series
from pyspark.pandas.utils import SPARK_CONF_ARROW_ENABLED
from pyspark.testing.sqlutils import ReusedSQLTestCase
from pyspark.errors import PySparkAssertionError
__all__ = ['assertPandasOnSparkEqual']

def _assert_pandas_equal(left: Union[pd.DataFrame, pd.Series, pd.Index], right: Union[pd.DataFrame, pd.Series, pd.Index], checkExact: bool):
    if False:
        while True:
            i = 10
    from pandas.core.dtypes.common import is_numeric_dtype
    from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        try:
            kwargs = dict(check_freq=False)
            assert_frame_equal(left, right, check_index_type='equiv' if len(left.index) > 0 else False, check_column_type='equiv' if len(left.columns) > 0 else False, check_exact=checkExact, **kwargs)
        except AssertionError:
            raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_DATAFRAME', message_parameters={'left': left.to_string(), 'left_dtype': str(left.dtypes), 'right': right.to_string(), 'right_dtype': str(right.dtypes)})
    elif isinstance(left, pd.Series) and isinstance(right, pd.Series):
        try:
            kwargs = dict(check_freq=False)
            assert_series_equal(left, right, check_index_type='equiv' if len(left.index) > 0 else False, check_exact=checkExact, **kwargs)
        except AssertionError:
            raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_SERIES', message_parameters={'left': left.to_string(), 'left_dtype': str(left.dtype), 'right': right.to_string(), 'right_dtype': str(right.dtype)})
    elif isinstance(left, pd.Index) and isinstance(right, pd.Index):
        try:
            assert_index_equal(left, right, check_exact=checkExact)
        except AssertionError:
            raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_INDEX', message_parameters={'left': left, 'left_dtype': str(left.dtype), 'right': right, 'right_dtype': str(right.dtype)})
    else:
        raise ValueError('Unexpected values: (%s, %s)' % (left, right))

def _assert_pandas_almost_equal(left: Union[pd.DataFrame, pd.Series, pd.Index], right: Union[pd.DataFrame, pd.Series, pd.Index], rtol: float=1e-05, atol: float=1e-08):
    if False:
        i = 10
        return i + 15
    '\n    This function checks if given pandas objects approximately same,\n    which means the conditions below:\n      - Both objects are nullable\n      - Compare decimals and floats, where two values a and b are approximately equal\n        if they satisfy the following formula:\n        absolute(a - b) <= (atol + rtol * absolute(b))\n        where rtol=1e-5 and atol=1e-8 by default\n    '

    def compare_vals_approx(val1, val2):
        if False:
            print('Hello World!')
        if isinstance(lval, (float, decimal.Decimal)) or isinstance(rval, (float, decimal.Decimal)):
            if abs(float(lval) - float(rval)) > atol + rtol * abs(float(rval)):
                return False
        elif val1 != val2:
            return False
        return True
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        if left.shape != right.shape:
            raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_DATAFRAME', message_parameters={'left': left.to_string(), 'left_dtype': str(left.dtypes), 'right': right.to_string(), 'right_dtype': str(right.dtypes)})
        for (lcol, rcol) in zip(left.columns, right.columns):
            if lcol != rcol:
                raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_DATAFRAME', message_parameters={'left': left.to_string(), 'left_dtype': str(left.dtypes), 'right': right.to_string(), 'right_dtype': str(right.dtypes)})
            for (lnull, rnull) in zip(left[lcol].isnull(), right[rcol].isnull()):
                if lnull != rnull:
                    raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_DATAFRAME', message_parameters={'left': left.to_string(), 'left_dtype': str(left.dtypes), 'right': right.to_string(), 'right_dtype': str(right.dtypes)})
            for (lval, rval) in zip(left[lcol].dropna(), right[rcol].dropna()):
                if not compare_vals_approx(lval, rval):
                    raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_DATAFRAME', message_parameters={'left': left.to_string(), 'left_dtype': str(left.dtypes), 'right': right.to_string(), 'right_dtype': str(right.dtypes)})
        if left.columns.names != right.columns.names:
            raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_DATAFRAME', message_parameters={'left': left.to_string(), 'left_dtype': str(left.dtypes), 'right': right.to_string(), 'right_dtype': str(right.dtypes)})
    elif isinstance(left, pd.Series) and isinstance(right, pd.Series):
        if left.name != right.name or len(left) != len(right):
            raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_SERIES', message_parameters={'left': left.to_string(), 'left_dtype': str(left.dtype), 'right': right.to_string(), 'right_dtype': str(right.dtype)})
        for (lnull, rnull) in zip(left.isnull(), right.isnull()):
            if lnull != rnull:
                raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_SERIES', message_parameters={'left': left.to_string(), 'left_dtype': str(left.dtype), 'right': right.to_string(), 'right_dtype': str(right.dtype)})
        for (lval, rval) in zip(left.dropna(), right.dropna()):
            if not compare_vals_approx(lval, rval):
                raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_SERIES', message_parameters={'left': left.to_string(), 'left_dtype': str(left.dtype), 'right': right.to_string(), 'right_dtype': str(right.dtype)})
    elif isinstance(left, pd.MultiIndex) and isinstance(right, pd.MultiIndex):
        if len(left) != len(right):
            raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_MULTIINDEX', message_parameters={'left': left, 'left_dtype': str(left.dtype), 'right': right, 'right_dtype': str(right.dtype)})
        for (lval, rval) in zip(left, right):
            if not compare_vals_approx(lval, rval):
                raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_MULTIINDEX', message_parameters={'left': left, 'left_dtype': str(left.dtype), 'right': right, 'right_dtype': str(right.dtype)})
    elif isinstance(left, pd.Index) and isinstance(right, pd.Index):
        if len(left) != len(right):
            raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_INDEX', message_parameters={'left': left, 'left_dtype': str(left.dtype), 'right': right, 'right_dtype': str(right.dtype)})
        for (lnull, rnull) in zip(left.isnull(), right.isnull()):
            if lnull != rnull:
                raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_INDEX', message_parameters={'left': left, 'left_dtype': str(left.dtype), 'right': right, 'right_dtype': str(right.dtype)})
        for (lval, rval) in zip(left.dropna(), right.dropna()):
            if not compare_vals_approx(lval, rval):
                raise PySparkAssertionError(error_class='DIFFERENT_PANDAS_INDEX', message_parameters={'left': left, 'left_dtype': str(left.dtype), 'right': right, 'right_dtype': str(right.dtype)})
    elif not isinstance(left, (pd.DataFrame, pd.Series, pd.Index)):
        raise PySparkAssertionError(error_class='INVALID_TYPE_DF_EQUALITY_ARG', message_parameters={'expected_type': f'{pd.DataFrame.__name__}, {pd.Series.__name__}, {pd.Index.__name__}, ', 'arg_name': 'left', 'actual_type': type(left)})
    elif not isinstance(right, (pd.DataFrame, pd.Series, pd.Index)):
        raise PySparkAssertionError(error_class='INVALID_TYPE_DF_EQUALITY_ARG', message_parameters={'expected_type': f'{pd.DataFrame.__name__}, {pd.Series.__name__}, {pd.Index.__name__}, ', 'arg_name': 'right', 'actual_type': type(right)})

def assertPandasOnSparkEqual(actual: Union[DataFrame, Series, Index], expected: Union[DataFrame, pd.DataFrame, Series, pd.Series, Index, pd.Index], checkExact: bool=True, almost: bool=False, rtol: float=1e-05, atol: float=1e-08, checkRowOrder: bool=True):
    if False:
        return 10
    "\n    A util function to assert equality between actual (pandas-on-Spark object) and expected\n    (pandas-on-Spark or pandas object).\n\n    .. versionadded:: 3.5.0\n\n    .. deprecated:: 3.5.1\n        `assertPandasOnSparkEqual` will be removed in Spark 4.0.0.\n        Use `ps.testing.assert_frame_equal`, `ps.testing.assert_series_equal`\n        and `ps.testing.assert_index_equal` instead.\n\n    Parameters\n    ----------\n    actual: pandas-on-Spark DataFrame, Series, or Index\n        The object that is being compared or tested.\n    expected: pandas-on-Spark or pandas DataFrame, Series, or Index\n        The expected object, for comparison with the actual result.\n    checkExact: bool, optional\n        A flag indicating whether to compare exact equality.\n        If set to 'True' (default), the data is compared exactly.\n        If set to 'False', the data is compared less precisely, following pandas assert_frame_equal\n        approximate comparison (see documentation for more details).\n    almost: bool, optional\n        A flag indicating whether to use unittest `assertAlmostEqual` or `assertEqual`.\n        If set to 'True', the comparison is delegated to `unittest`'s `assertAlmostEqual`\n        (see documentation for more details).\n        If set to 'False' (default), the data is compared exactly with `unittest`'s\n        `assertEqual`.\n    rtol : float, optional\n        The relative tolerance, used in asserting almost equality for float values in actual\n        and expected. Set to 1e-5 by default. (See Notes)\n    atol : float, optional\n        The absolute tolerance, used in asserting almost equality for float values in actual\n        and expected. Set to 1e-8 by default. (See Notes)\n    checkRowOrder : bool, optional\n        A flag indicating whether the order of rows should be considered in the comparison.\n        If set to `False`, the row order is not taken into account.\n        If set to `True` (default), the order of rows will be checked during comparison.\n        (See Notes)\n\n    Notes\n    -----\n    For `checkRowOrder`, note that pandas-on-Spark DataFrame ordering is non-deterministic, unless\n    explicitly sorted.\n\n    When `almost` is set to True, approximate equality will be asserted, where two values\n    a and b are approximately equal if they satisfy the following formula:\n\n    ``absolute(a - b) <= (atol + rtol * absolute(b))``.\n\n    Examples\n    --------\n    >>> import pyspark.pandas as ps\n    >>> psdf1 = ps.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})\n    >>> psdf2 = ps.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})\n    >>> assertPandasOnSparkEqual(psdf1, psdf2)  # pass, ps.DataFrames are equal\n    >>> s1 = ps.Series([212.32, 100.0001])\n    >>> s2 = ps.Series([212.32, 100.0])\n    >>> assertPandasOnSparkEqual(s1, s2, checkExact=False)  # pass, ps.Series are approx equal\n    >>> s1 = ps.Index([212.300001, 100.000])\n    >>> s2 = ps.Index([212.3, 100.0001])\n    >>> assertPandasOnSparkEqual(s1, s2, almost=True)  # pass, ps.Index obj are almost equal\n    "
    warnings.warn('`assertPandasOnSparkEqual` will be removed in Spark 4.0.0. Use `ps.testing.assert_frame_equal`, `ps.testing.assert_series_equal` and `ps.testing.assert_index_equal` instead.', FutureWarning)
    if actual is None and expected is None:
        return True
    elif actual is None or expected is None:
        return False
    if not isinstance(actual, (DataFrame, Series, Index)):
        raise PySparkAssertionError(error_class='INVALID_TYPE_DF_EQUALITY_ARG', message_parameters={'expected_type': f'{DataFrame.__name__}, {Series.__name__}, {Index.__name__}', 'arg_name': 'actual', 'actual_type': type(actual)})
    elif not isinstance(expected, (DataFrame, pd.DataFrame, Series, pd.Series, Index, pd.Index)):
        raise PySparkAssertionError(error_class='INVALID_TYPE_DF_EQUALITY_ARG', message_parameters={'expected_type': f'{DataFrame.__name__}, {pd.DataFrame.__name__}, {Series.__name__}, {pd.Series.__name__}, {Index.__name__}{pd.Index.__name__}, ', 'arg_name': 'expected', 'actual_type': type(expected)})
    else:
        if not isinstance(actual, (pd.DataFrame, pd.Index, pd.Series)):
            actual = actual.to_pandas()
        if not isinstance(expected, (pd.DataFrame, pd.Index, pd.Series)):
            expected = expected.to_pandas()
        if not checkRowOrder:
            if isinstance(actual, pd.DataFrame) and len(actual.columns) > 0:
                actual = actual.sort_values(by=actual.columns[0], ignore_index=True)
            if isinstance(expected, pd.DataFrame) and len(expected.columns) > 0:
                expected = expected.sort_values(by=expected.columns[0], ignore_index=True)
        if almost:
            _assert_pandas_almost_equal(actual, expected, rtol=rtol, atol=atol)
        else:
            _assert_pandas_equal(actual, expected, checkExact=checkExact)

class PandasOnSparkTestUtils:

    def convert_str_to_lambda(self, func: str):
        if False:
            i = 10
            return i + 15
        '\n        This function converts `func` str to lambda call\n        '
        return lambda x: getattr(x, func)()

    def assertPandasEqual(self, left: Any, right: Any, check_exact: bool=True):
        if False:
            for i in range(10):
                print('nop')
        _assert_pandas_equal(left, right, check_exact)

    def assertPandasAlmostEqual(self, left: Any, right: Any, rtol: float=1e-05, atol: float=1e-08):
        if False:
            while True:
                i = 10
        _assert_pandas_almost_equal(left, right, rtol=rtol, atol=atol)

    def assert_eq(self, left: Any, right: Any, check_exact: bool=True, almost: bool=False, rtol: float=1e-05, atol: float=1e-08, check_row_order: bool=True):
        if False:
            return 10
        "\n        Asserts if two arbitrary objects are equal or not. If given objects are Koalas DataFrame\n        or Series, they are converted into pandas' and compared.\n\n        :param left: object to compare\n        :param right: object to compare\n        :param check_exact: if this is False, the comparison is done less precisely.\n        :param almost: if this is enabled, the comparison asserts approximate equality\n            for float and decimal values, where two values a and b are approximately equal\n            if they satisfy the following formula:\n            absolute(a - b) <= (atol + rtol * absolute(b))\n        :param rtol: The relative tolerance, used in asserting approximate equality for\n            float values. Set to 1e-5 by default.\n        :param atol: The absolute tolerance, used in asserting approximate equality for\n            float values in actual and expected. Set to 1e-8 by default.\n        :param check_row_order: A flag indicating whether the order of rows should be considered\n            in the comparison. If set to False, row order will be ignored.\n        "
        import pandas as pd
        from pandas.api.types import is_list_like
        if isinstance(left, (ps.DataFrame, ps.Series, ps.Index)):
            return assertPandasOnSparkEqual(left, right, checkExact=check_exact, almost=almost, rtol=rtol, atol=atol, checkRowOrder=check_row_order)
        lobj = self._to_pandas(left)
        robj = self._to_pandas(right)
        if isinstance(lobj, (pd.DataFrame, pd.Series, pd.Index)):
            if almost:
                _assert_pandas_almost_equal(lobj, robj, rtol=rtol, atol=atol)
            else:
                _assert_pandas_equal(lobj, robj, checkExact=check_exact)
        elif is_list_like(lobj) and is_list_like(robj):
            self.assertTrue(len(left) == len(right))
            for (litem, ritem) in zip(left, right):
                self.assert_eq(litem, ritem, check_exact=check_exact, almost=almost)
        elif (lobj is not None and pd.isna(lobj)) and (robj is not None and pd.isna(robj)):
            pass
        elif almost:
            self.assertAlmostEqual(lobj, robj)
        else:
            self.assertEqual(lobj, robj)

    @staticmethod
    def _to_pandas(obj: Any):
        if False:
            i = 10
            return i + 15
        if isinstance(obj, (DataFrame, Series, Index)):
            return obj.to_pandas()
        else:
            return obj

class PandasOnSparkTestCase(ReusedSQLTestCase, PandasOnSparkTestUtils):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super(PandasOnSparkTestCase, cls).setUpClass()
        cls.spark.conf.set(SPARK_CONF_ARROW_ENABLED, True)

class TestUtils:

    @contextmanager
    def temp_dir(self):
        if False:
            i = 10
            return i + 15
        tmp = tempfile.mkdtemp()
        try:
            yield tmp
        finally:
            shutil.rmtree(tmp)

    @contextmanager
    def temp_file(self):
        if False:
            print('Hello World!')
        with self.temp_dir() as tmp:
            yield tempfile.mkstemp(dir=tmp)[1]

class ComparisonTestBase(PandasOnSparkTestCase):

    @property
    def psdf(self):
        if False:
            for i in range(10):
                print('nop')
        return ps.from_pandas(self.pdf)

    @property
    def pdf(self):
        if False:
            print('Hello World!')
        return self.psdf.to_pandas()

def compare_both(f=None, almost=True):
    if False:
        while True:
            i = 10
    if f is None:
        return functools.partial(compare_both, almost=almost)
    elif isinstance(f, bool):
        return functools.partial(compare_both, almost=f)

    @functools.wraps(f)
    def wrapped(self):
        if False:
            for i in range(10):
                print('nop')
        if almost:
            compare = self.assertPandasAlmostEqual
        else:
            compare = self.assertPandasEqual
        for (result_pandas, result_spark) in zip(f(self, self.pdf), f(self, self.psdf)):
            compare(result_pandas, result_spark.to_pandas())
    return wrapped

@contextmanager
def assert_produces_warning(expected_warning=Warning, filter_level='always', check_stacklevel=True, raise_on_extra_warnings=True):
    if False:
        print('Hello World!')
    '\n    Context manager for running code expected to either raise a specific\n    warning, or not raise any warnings. Verifies that the code raises the\n    expected warning, and that it does not raise any other unexpected\n    warnings. It is basically a wrapper around ``warnings.catch_warnings``.\n\n    Notes\n    -----\n    Replicated from pandas/_testing/_warnings.py.\n\n    Parameters\n    ----------\n    expected_warning : {Warning, False, None}, default Warning\n        The type of Exception raised. ``exception.Warning`` is the base\n        class for all warnings. To check that no warning is returned,\n        specify ``False`` or ``None``.\n    filter_level : str or None, default "always"\n        Specifies whether warnings are ignored, displayed, or turned\n        into errors.\n        Valid values are:\n        * "error" - turns matching warnings into exceptions\n        * "ignore" - discard the warning\n        * "always" - always emit a warning\n        * "default" - print the warning the first time it is generated\n          from each location\n        * "module" - print the warning the first time it is generated\n          from each module\n        * "once" - print the warning the first time it is generated\n    check_stacklevel : bool, default True\n        If True, displays the line that called the function containing\n        the warning to show were the function is called. Otherwise, the\n        line that implements the function is displayed.\n    raise_on_extra_warnings : bool, default True\n        Whether extra warnings not of the type `expected_warning` should\n        cause the test to fail.\n\n    Examples\n    --------\n    >>> import warnings\n    >>> with assert_produces_warning():\n    ...     warnings.warn(UserWarning())\n    ...\n    >>> with assert_produces_warning(False): # doctest: +SKIP\n    ...     warnings.warn(RuntimeWarning())\n    ...\n    Traceback (most recent call last):\n        ...\n    AssertionError: Caused unexpected warning(s): [\'RuntimeWarning\'].\n    >>> with assert_produces_warning(UserWarning): # doctest: +SKIP\n    ...     warnings.warn(RuntimeWarning())\n    Traceback (most recent call last):\n        ...\n    AssertionError: Did not see expected warning of class \'UserWarning\'\n    ..warn:: This is *not* thread-safe.\n    '
    __tracebackhide__ = True
    with warnings.catch_warnings(record=True) as w:
        saw_warning = False
        warnings.simplefilter(filter_level)
        yield w
        extra_warnings = []
        for actual_warning in w:
            if expected_warning and issubclass(actual_warning.category, expected_warning):
                saw_warning = True
                if check_stacklevel and issubclass(actual_warning.category, (FutureWarning, DeprecationWarning)):
                    from inspect import getframeinfo, stack
                    caller = getframeinfo(stack()[2][0])
                    msg = ('Warning not set with correct stacklevel. ', 'File where warning is raised: {} != '.format(actual_warning.filename), '{}. Warning message: {}'.format(caller.filename, actual_warning.message))
                    assert actual_warning.filename == caller.filename, msg
            else:
                extra_warnings.append((actual_warning.category.__name__, actual_warning.message, actual_warning.filename, actual_warning.lineno))
        if expected_warning:
            msg = 'Did not see expected warning of class {}'.format(repr(expected_warning.__name__))
            assert saw_warning, msg
        if raise_on_extra_warnings and extra_warnings:
            raise AssertionError('Caused unexpected warning(s): {}'.format(repr(extra_warnings)))