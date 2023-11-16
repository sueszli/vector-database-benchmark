import re
import unittest
import warnings
from typing import Dict
import numpy as np
import pandas as pd
import pytest
from parameterized import parameterized
import apache_beam as beam
from apache_beam.dataframe import expressions
from apache_beam.dataframe import frame_base
from apache_beam.dataframe import frames
from apache_beam.dataframe.convert import to_dataframe
from apache_beam.runners.interactive import interactive_beam as ib
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive.interactive_runner import InteractiveRunner
from apache_beam.runners.interactive.testing.mock_env import isolated_env
PD_VERSION = tuple(map(int, pd.__version__.split('.')[0:2]))
GROUPBY_DF = pd.DataFrame({'group': ['a' if i % 5 == 0 or i % 3 == 0 else 'b' for i in range(100)], 'foo': [None if i % 11 == 0 else i for i in range(100)], 'bar': [None if i % 7 == 0 else 99 - i for i in range(100)], 'baz': [None if i % 13 == 0 else i * 2 for i in range(100)], 'bool': [i % 17 == 0 for i in range(100)], 'str': [str(i) for i in range(100)]})
if PD_VERSION < (2, 0):
    pytestmark = pytest.mark.filterwarnings('ignore::FutureWarning')

def _get_deferred_args(*args):
    if False:
        return 10
    return [frame_base.DeferredFrame.wrap(expressions.ConstantExpression(arg, arg[0:0])) for arg in args]

class _AbstractFrameTest(unittest.TestCase):
    """Test sub-class with utilities for verifying DataFrame operations."""

    def _run_error_test(self, func, *args, construction_time=True, distributed=True):
        if False:
            while True:
                i = 10
        'Verify that func(*args) raises the same exception in pandas and in Beam.\n\n    Note that by default this only checks for exceptions that the Beam DataFrame\n    API raises during expression generation (i.e. construction time).\n    Exceptions raised while the pipeline is executing are less helpful, but\n    are sometimes unavoidable (e.g. data validation exceptions), to check for\n    these exceptions use construction_time=False.'
        deferred_args = _get_deferred_args(*args)
        try:
            expected = func(*args)
        except Exception as e:
            expected_error = e
        else:
            raise AssertionError(f'Expected an error, but executing with pandas successfully returned:\n{expected}')
        if construction_time:
            try:
                _ = func(*deferred_args)._expr
            except Exception as e:
                actual = e
            else:
                raise AssertionError(f'Expected an error:\n{expected_error}\nbut Beam successfully generated an expression.')
        else:
            expr = func(*deferred_args)._expr
            session_type = expressions.PartitioningSession if distributed else expressions.Session
            try:
                result = session_type({}).evaluate(expr)
            except Exception as e:
                actual = e
            else:
                raise AssertionError(f'Expected an error:\n{expected_error}\nbut Beam successfully Computed the result:\n{result}.')
        if not isinstance(actual, type(expected_error)) or str(expected_error) not in str(actual):
            raise AssertionError(f'Expected {expected_error!r} to be raised, but got {actual!r}') from actual

    def _run_inplace_test(self, func, arg, **kwargs):
        if False:
            return 10
        'Verify an inplace operation performed by func.\n\n    Checks that func performs the same inplace operation on arg, in pandas and\n    in Beam.'

        def wrapper(df):
            if False:
                i = 10
                return i + 15
            df = df.copy()
            func(df)
            return df
        self._run_test(wrapper, arg, **kwargs)

    def _run_test(self, func, *args, distributed=True, nonparallel=False, check_proxy=True, lenient_dtype_check=False):
        if False:
            for i in range(10):
                print('nop')
        'Verify that func(*args) produces the same result in pandas and in Beam.\n\n    Args:\n        distributed (bool): Whether or not to use PartitioningSession to\n            simulate parallel execution.\n        nonparallel (bool): Whether or not this function contains a\n            non-parallelizable operation. If True, the expression will be\n            generated twice, once outside of an allow_non_parallel_operations\n            block (to verify NonParallelOperation is raised), and again inside\n            of an allow_non_parallel_operations block to actually generate an\n            expression to verify.\n        check_proxy (bool): Whether or not to check that the proxy of the\n            generated expression matches the actual result, defaults to True.\n            This option should NOT be set to False in tests added for new\n            operations if at all possible. Instead make sure the new operation\n            produces the correct proxy. This flag only exists as an escape hatch\n            until existing failures can be addressed\n            (https://github.com/apache/beam/issues/20926).\n        lenient_dtype_check (bool): Whether or not to check that numeric columns\n            are still numeric between actual and proxy. i.e. verify that they\n            are at least int64 or float64, and not necessarily have the exact\n            same dtype. This may need to be set to True for some non-deferred\n            operations, where the dtype of the values in the proxy are not known\n            ahead of time, causing int64 to float64 coercion issues.\n    '
        expected = func(*args)
        deferred_args = _get_deferred_args(*args)
        if nonparallel:
            with self.assertRaises(expressions.NonParallelOperation) as raised:
                func(*deferred_args)
            if raised.exception.msg.startswith('Encountered non-parallelizable form of'):
                raise AssertionError('Default NonParallelOperation raised, please specify a reason in the Singleton() partitioning requirement for this operation.') from raised.exception
            with beam.dataframe.allow_non_parallel_operations():
                expr = func(*deferred_args)._expr
        else:
            expr = func(*deferred_args)._expr
        session_type = expressions.PartitioningSession if distributed else expressions.Session
        actual = session_type({}).evaluate(expr)
        if isinstance(expected, pd.core.generic.NDFrame):
            if distributed:
                if expected.index.is_unique:
                    expected = expected.sort_index()
                    actual = actual.sort_index()
                elif isinstance(expected, pd.Series):
                    expected = expected.sort_values()
                    actual = actual.sort_values()
                else:
                    expected = expected.sort_values(list(expected.columns))
                    actual = actual.sort_values(list(actual.columns))
            if isinstance(expected, pd.Series):
                if lenient_dtype_check:
                    pd.testing.assert_series_equal(expected.astype('Float64'), actual.astype('Float64'))
                else:
                    pd.testing.assert_series_equal(expected, actual)
            elif isinstance(expected, pd.DataFrame):
                if lenient_dtype_check:
                    pd.testing.assert_frame_equal(expected.astype('Float64'), actual.astype('Float64'))
                else:
                    pd.testing.assert_frame_equal(expected, actual)
            else:
                raise ValueError(f'Expected value is a {type(expected)},not a Series or DataFrame.')
        else:
            if isinstance(expected, float):
                if np.isnan(expected):
                    cmp = np.isnan
                else:
                    cmp = lambda x: np.isclose(expected, x)
            else:
                cmp = lambda x: x == expected
            self.assertTrue(cmp(actual), 'Expected:\n\n%r\n\nActual:\n\n%r' % (expected, actual))
        if check_proxy:
            proxy = expr.proxy()
            if type(actual) in (np.float32, np.float64):
                self.assertTrue(type(actual) == type(proxy) or np.isnan(proxy))
            else:
                self.assertEqual(type(actual), type(proxy))
            if isinstance(expected, pd.core.generic.NDFrame):
                if isinstance(expected, pd.Series):
                    if lenient_dtype_check:
                        self.assertEqual(actual.astype('Float64').dtype, proxy.astype('Float64').dtype)
                    else:
                        self.assertEqual(actual.dtype, proxy.dtype)
                    self.assertEqual(actual.name, proxy.name)
                elif isinstance(expected, pd.DataFrame):
                    if lenient_dtype_check:
                        pd.testing.assert_series_equal(actual.astype('Float64').dtypes, proxy.astype('Float64').dtypes)
                    else:
                        pd.testing.assert_series_equal(actual.dtypes, proxy.dtypes)
                else:
                    raise ValueError(f'Expected value is a {type(expected)},not a Series or DataFrame.')
                self.assertEqual(actual.index.names, proxy.index.names)
                for i in range(actual.index.nlevels):
                    if lenient_dtype_check:
                        self.assertEqual(actual.astype('Float64').index.get_level_values(i).dtype, proxy.astype('Float64').index.get_level_values(i).dtype)
                    else:
                        self.assertEqual(actual.index.get_level_values(i).dtype, proxy.index.get_level_values(i).dtype)

class DeferredFrameTest(_AbstractFrameTest):
    """Miscellaneous tessts for DataFrame operations."""

    def test_series_arithmetic(self):
        if False:
            return 10
        a = pd.Series([1, 2, 3])
        b = pd.Series([100, 200, 300])
        self._run_test(lambda a, b: a - 2 * b, a, b)
        self._run_test(lambda a, b: a.subtract(2).multiply(b).divide(a), a, b)

    def test_dataframe_arithmetic(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [100, 200, 300]})
        df2 = pd.DataFrame({'a': [3000, 1000, 2000], 'b': [7, 11, 13]})
        self._run_test(lambda df, df2: df - 2 * df2, df, df2)
        self._run_test(lambda df, df2: df.subtract(2).multiply(df2).divide(df), df, df2)

    @unittest.skipIf(PD_VERSION < (1, 3), 'dropna=False is new in pandas 1.3')
    def test_value_counts_dropna_false(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'first_name': ['John', 'Anne', 'John', 'Beth'], 'middle_name': ['Smith', pd.NA, pd.NA, 'Louise']})
        with self.assertRaises(NotImplementedError):
            self._run_test(lambda df: df.value_counts(dropna=False), df)

    def test_get_column(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'], 'Speed': [380.0, 370.0, 24.0, 26.0]})
        self._run_test(lambda df: df['Animal'], df)
        self._run_test(lambda df: df.Speed, df)
        self._run_test(lambda df: df.get('Animal'), df)
        self._run_test(lambda df: df.get('FOO', df.Animal), df)

    def test_series_xs(self):
        if False:
            print('Hello World!')
        d = {'num_legs': [4, 4, 2, 2], 'num_wings': [0, 0, 2, 2], 'class': ['mammal', 'mammal', 'mammal', 'bird'], 'animal': ['cat', 'dog', 'bat', 'penguin'], 'locomotion': ['walks', 'walks', 'flies', 'walks']}
        df = pd.DataFrame(data=d)
        df = df.set_index(['class', 'animal', 'locomotion'])
        self._run_test(lambda df: df.num_legs.xs('mammal'), df)
        self._run_test(lambda df: df.num_legs.xs(('mammal', 'dog')), df)
        self._run_test(lambda df: df.num_legs.xs('cat', level=1), df)
        self._run_test(lambda df: df.num_legs.xs(('bird', 'walks'), level=[0, 'locomotion']), df)

    def test_dataframe_xs(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame(np.array([['state', 'day1', 12], ['state', 'day1', 1], ['state', 'day2', 14], ['county', 'day1', 9]]), columns=['provider', 'time', 'value'])
        self._run_test(lambda df: df.xs('state'), df.set_index(['provider']))
        self._run_test(lambda df: df.xs('state'), df.set_index(['provider', 'time']))

    def test_set_column(self):
        if False:
            return 10

        def new_column(df):
            if False:
                for i in range(10):
                    print('nop')
            df['NewCol'] = df['Speed']
        df = pd.DataFrame({'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'], 'Speed': [380.0, 370.0, 24.0, 26.0]})
        self._run_inplace_test(new_column, df)

    def test_set_column_from_index(self):
        if False:
            return 10

        def new_column(df):
            if False:
                for i in range(10):
                    print('nop')
            df['NewCol'] = df.index
        df = pd.DataFrame({'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'], 'Speed': [380.0, 370.0, 24.0, 26.0]})
        self._run_inplace_test(new_column, df)

    def test_tz_localize_ambiguous_series(self):
        if False:
            while True:
                i = 10
        s = pd.Series(range(3), index=pd.DatetimeIndex(['2018-10-28 01:20:00', '2018-10-28 02:36:00', '2018-10-28 03:46:00']))
        ambiguous = pd.Series([True, True, False], index=s.index)
        self._run_test(lambda s, ambiguous: s.tz_localize('CET', ambiguous=ambiguous), s, ambiguous)

    def test_tz_convert(self):
        if False:
            print('Hello World!')
        s = pd.Series(range(3), index=pd.DatetimeIndex(['2018-10-27 01:20:00', '2018-10-27 02:36:00', '2018-10-27 03:46:00'], tz='Europe/Berlin'))
        self._run_test(lambda s: s.tz_convert('America/Los_Angeles'), s)

    def test_sort_index_columns(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'c': range(10), 'a': range(10), 'b': range(10), np.nan: range(10)})
        self._run_test(lambda df: df.sort_index(axis=1), df)
        self._run_test(lambda df: df.sort_index(axis=1, ascending=False), df)
        self._run_test(lambda df: df.sort_index(axis=1, na_position='first'), df)

    def test_where_callable_args(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame(np.arange(10, dtype=np.int64).reshape(-1, 2), columns=['A', 'B'])
        self._run_test(lambda df: df.where(lambda df: df % 2 == 0, lambda df: df * 10), df)

    def test_where_concrete_args(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame(np.arange(10, dtype=np.int64).reshape(-1, 2), columns=['A', 'B'])
        self._run_test(lambda df: df.where(df % 2 == 0, pd.Series({'A': 123, 'B': 456}), axis=1), df)

    def test_combine_dataframe(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'A': [0, 0], 'B': [4, 4]})
        df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})
        take_smaller = lambda s1, s2: s1 if s1.sum() < s2.sum() else s2
        self._run_test(lambda df, df2: df.combine(df2, take_smaller), df, df2, nonparallel=True)

    def test_combine_dataframe_fill(self):
        if False:
            print('Hello World!')
        df1 = pd.DataFrame({'A': [0, 0], 'B': [None, 4]})
        df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})
        take_smaller = lambda s1, s2: s1 if s1.sum() < s2.sum() else s2
        self._run_test(lambda df1, df2: df1.combine(df2, take_smaller, fill_value=-5), df1, df2, nonparallel=True)

    def test_combine_Series(self):
        if False:
            print('Hello World!')
        s1 = pd.Series({'falcon': 330.0, 'eagle': 160.0})
        s2 = pd.Series({'falcon': 345.0, 'eagle': 200.0, 'duck': 30.0})
        self._run_test(lambda s1, s2: s1.combine(s2, max), s1, s2, nonparallel=True, check_proxy=False)

    def test_combine_first_dataframe(self):
        if False:
            return 10
        df1 = pd.DataFrame({'A': [None, 0], 'B': [None, 4]})
        df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})
        self._run_test(lambda df1, df2: df1.combine_first(df2), df1, df2)

    def test_combine_first_series(self):
        if False:
            return 10
        s1 = pd.Series([1, np.nan])
        s2 = pd.Series([3, 4])
        self._run_test(lambda s1, s2: s1.combine_first(s2), s1, s2)

    def test_add_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
        s = pd.Series([1, 2, 3, 4])
        self._run_test(lambda df: df.add_prefix('col_'), df)
        self._run_test(lambda s: s.add_prefix('col_'), s)

    def test_add_suffix(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
        s = pd.Series([1, 2, 3, 4])
        self._run_test(lambda df: df.add_suffix('_col'), df)
        self._run_test(lambda s: s.add_prefix('_col'), s)

    def test_set_index(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'index1': reversed(range(20)), 'index2': np.roll(range(20), 5), 'values': [chr(ord('a') + i) * i for i in range(20)]})
        self._run_test(lambda df: df.set_index(['index1', 'index2']), df)
        self._run_test(lambda df: df.set_index(['index1', 'index2'], drop=True), df)
        self._run_test(lambda df: df.set_index('values'), df)
        self._run_error_test(lambda df: df.set_index('bad'), df)
        self._run_error_test(lambda df: df.set_index(['index2', 'bad', 'really_bad']), df)

    def test_set_axis(self):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['X', 'Y', 'Z'])
        self._run_test(lambda df: df.set_axis(['I', 'II'], axis='columns'), df)
        self._run_test(lambda df: df.set_axis([0, 1], axis=1), df)
        self._run_inplace_test(lambda df: df.set_axis(['i', 'ii'], axis='columns'), df)
        with self.assertRaises(NotImplementedError):
            self._run_test(lambda df: df.set_axis(['a', 'b', 'c'], axis='index'), df)
            self._run_test(lambda df: df.set_axis([0, 1, 2], axis=0), df)

    def test_series_set_axis(self):
        if False:
            return 10
        s = pd.Series(list(range(3)), index=['X', 'Y', 'Z'])
        with self.assertRaises(NotImplementedError):
            self._run_test(lambda s: s.set_axis(['a', 'b', 'c']), s)
            self._run_test(lambda s: s.set_axis([1, 2, 3]), s)

    def test_series_drop_ignore_errors(self):
        if False:
            print('Hello World!')
        midx = pd.MultiIndex(levels=[['lama', 'cow', 'falcon'], ['speed', 'weight', 'length']], codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        s = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        self._run_test(lambda s: s.drop('lama', level=0, errors='ignore'), s)
        self._run_test(lambda s: s.drop(('cow', 'speed'), errors='ignore'), s)
        self._run_test(lambda s: s.drop('falcon', level=0, errors='ignore'), s)

    def test_dataframe_drop_ignore_errors(self):
        if False:
            for i in range(10):
                print('nop')
        midx = pd.MultiIndex(levels=[['lama', 'cow', 'falcon'], ['speed', 'weight', 'length']], codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        df = pd.DataFrame(index=midx, columns=['big', 'small'], data=[[45, 30], [200, 100], [1.5, 1], [30, 20], [250, 150], [1.5, 0.8], [320, 250], [1, 0.8], [0.3, 0.2]])
        self._run_test(lambda df: df.drop(index='lama', level=0, errors='ignore'), df)
        self._run_test(lambda df: df.drop(index=('cow', 'speed'), errors='ignore'), df)
        self._run_test(lambda df: df.drop(index='falcon', level=0, errors='ignore'), df)
        self._run_test(lambda df: df.drop(index='cow', columns='small', errors='ignore'), df)

    def test_merge(self):
        if False:
            return 10
        df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
        df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
        self._run_test(lambda df1, df2: df1.merge(df2, left_on='lkey', right_on='rkey').rename(index=lambda x: '*'), df1, df2, nonparallel=True, check_proxy=False)
        self._run_test(lambda df1, df2: df1.merge(df2, left_on='lkey', right_on='rkey', suffixes=('_left', '_right')).rename(index=lambda x: '*'), df1, df2, nonparallel=True, check_proxy=False)

    def test_merge_left_join(self):
        if False:
            return 10
        df1 = pd.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2]})
        df2 = pd.DataFrame({'a': ['foo', 'baz'], 'c': [3, 4]})
        self._run_test(lambda df1, df2: df1.merge(df2, how='left', on='a').rename(index=lambda x: '*'), df1, df2, nonparallel=True, check_proxy=False)

    def test_merge_on_index(self):
        if False:
            return 10
        df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]}).set_index('lkey')
        df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]}).set_index('rkey')
        self._run_test(lambda df1, df2: df1.merge(df2, left_index=True, right_index=True), df1, df2, check_proxy=False)

    def test_merge_same_key(self):
        if False:
            return 10
        df1 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
        df2 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
        self._run_test(lambda df1, df2: df1.merge(df2, on='key').rename(index=lambda x: '*'), df1, df2, nonparallel=True, check_proxy=False)
        self._run_test(lambda df1, df2: df1.merge(df2, on='key', suffixes=('_left', '_right')).rename(index=lambda x: '*'), df1, df2, nonparallel=True, check_proxy=False)

    def test_merge_same_key_doctest(self):
        if False:
            while True:
                i = 10
        df1 = pd.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2]})
        df2 = pd.DataFrame({'a': ['foo', 'baz'], 'c': [3, 4]})
        self._run_test(lambda df1, df2: df1.merge(df2, how='left', on='a').rename(index=lambda x: '*'), df1, df2, nonparallel=True, check_proxy=False)
        self._run_test(lambda df1, df2: df1.merge(df2, how='left').rename(index=lambda x: '*'), df1, df2, nonparallel=True, check_proxy=False)

    def test_merge_same_key_suffix_collision(self):
        if False:
            for i in range(10):
                print('nop')
        df1 = pd.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2], 'a_lsuffix': [5, 6]})
        df2 = pd.DataFrame({'a': ['foo', 'baz'], 'c': [3, 4], 'a_rsuffix': [7, 8]})
        self._run_test(lambda df1, df2: df1.merge(df2, how='left', on='a', suffixes=('_lsuffix', '_rsuffix')).rename(index=lambda x: '*'), df1, df2, nonparallel=True, check_proxy=False)
        self._run_test(lambda df1, df2: df1.merge(df2, how='left', suffixes=('_lsuffix', '_rsuffix')).rename(index=lambda x: '*'), df1, df2, nonparallel=True, check_proxy=False)

    def test_swaplevel(self):
        if False:
            return 10
        df = pd.DataFrame({'Grade': ['A', 'B', 'A', 'C']}, index=[['Final exam', 'Final exam', 'Coursework', 'Coursework'], ['History', 'Geography', 'History', 'Geography'], ['January', 'February', 'March', 'April']])
        self._run_test(lambda df: df.swaplevel(), df)

    def test_value_counts_with_nans(self):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'num_legs': [2, 4, 4, 6, np.nan, np.nan], 'num_wings': [2, 0, 0, 0, np.nan, 2]}, index=['falcon', 'dog', 'cat', 'ant', 'car', 'plane'])
        self._run_test(lambda df: df.value_counts(), df)
        self._run_test(lambda df: df.value_counts(normalize=True), df)
        self._run_test(lambda df: df.value_counts('num_wings'), df)
        if PD_VERSION >= (1, 3):
            with self.assertRaises(NotImplementedError):
                self._run_test(lambda df: df.value_counts(dropna=False), df)
        self._run_test(lambda df: df.num_wings.value_counts(), df)
        self._run_test(lambda df: df.num_wings.value_counts(normalize=True), df)
        self._run_test(lambda df: df.num_wings.value_counts(dropna=False), df)
        for normalize in (True, False):
            for dropna in (True, False):
                self._run_test(lambda df, dropna=dropna, normalize=normalize: df.num_wings.value_counts(dropna=dropna, normalize=normalize), df)

    def test_value_counts_does_not_support_sort(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'num_legs': [2, 4, 4, 6, np.nan, np.nan], 'num_wings': [2, 0, 0, 0, np.nan, 2]}, index=['falcon', 'dog', 'cat', 'ant', 'car', 'plane'])
        with self.assertRaisesRegex(frame_base.WontImplementError, 'value_counts\\(sort\\=True\\)'):
            self._run_test(lambda df: df.value_counts(sort=True), df)
        with self.assertRaisesRegex(frame_base.WontImplementError, 'value_counts\\(sort\\=True\\)'):
            self._run_test(lambda df: df.num_wings.value_counts(sort=True), df)

    def test_series_getitem(self):
        if False:
            while True:
                i = 10
        s = pd.Series([x ** 2 for x in range(10)])
        self._run_test(lambda s: s[...], s)
        self._run_test(lambda s: s[:], s)
        self._run_test(lambda s: s[s < 10], s)
        self._run_test(lambda s: s[lambda s: s < 10], s)
        s.index = s.index.map(float)
        self._run_test(lambda s: s[1.5:6], s)

    def test_series_truncate(self):
        if False:
            return 10
        s = pd.Series(['a', 'b', 'c', 'd', 'e', 'f'])
        self._run_test(lambda s: s.truncate(before=1, after=3), s)

    def test_dataframe_truncate(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'C': list('abcde'), 'B': list('fghij'), 'A': list('klmno')}, index=[1, 2, 3, 4, 5])
        self._run_test(lambda df: df.truncate(before=1, after=3), df)
        self._run_test(lambda df: df.truncate(before='A', after='B', axis=1), df)
        self._run_test(lambda df: df['A'].truncate(before=2, after=4), df)

    @parameterized.expand([(pd.Series(range(10)),), (pd.Series(list(range(100)) + [0]),), (pd.Series(list(range(100)) + [0]) / 100,), (pd.Series(['a', 'b', 'c', 'd']),), (pd.Series(['a', 'b', 'a', 'c', 'd']),)])
    def test_series_is_unique(self, series):
        if False:
            for i in range(10):
                print('nop')
        self._run_test(lambda s: s.is_unique, series)

    @parameterized.expand([(pd.Series(range(10)),), (pd.Series([1, 2, np.nan, 3, np.nan]),), (pd.Series(['a', 'b', 'c', 'd', 'e']),), (pd.Series(['a', 'b', None, 'c', None]),)])
    def test_series_hasnans(self, series):
        if False:
            while True:
                i = 10
        self._run_test(lambda s: s.hasnans, series)

    def test_dataframe_getitem(self):
        if False:
            return 10
        df = pd.DataFrame({'A': [x ** 2 for x in range(6)], 'B': list('abcdef')})
        self._run_test(lambda df: df['A'], df)
        self._run_test(lambda df: df[['A', 'B']], df)
        self._run_test(lambda df: df[:], df)
        self._run_test(lambda df: df[df.A < 10], df)
        df.index = df.index.map(float)
        self._run_test(lambda df: df[1.5:4], df)

    def test_loc(self):
        if False:
            i = 10
            return i + 15
        dates = pd.date_range('1/1/2000', periods=8)
        dates.freq = None
        df = pd.DataFrame(np.arange(32).reshape((8, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
        self._run_test(lambda df: df.loc[:], df)
        self._run_test(lambda df: df.loc[:, 'A'], df)
        self._run_test(lambda df: df.loc[:dates[3]], df)
        self._run_test(lambda df: df.loc[df.A > 10], df)
        self._run_test(lambda df: df.loc[lambda df: df.A > 10], df)
        self._run_test(lambda df: df.C.loc[df.A > 10], df)
        self._run_test(lambda df, s: df.loc[s.loc[1:3]], df, pd.Series(dates))

    @unittest.skipIf(PD_VERSION >= (2, 0), 'append removed in Pandas 2.0')
    def test_append_sort(self):
        if False:
            while True:
                i = 10
        df1 = pd.DataFrame({'int': [1, 2, 3], 'str': ['a', 'b', 'c']}, columns=['int', 'str'], index=[1, 3, 5])
        df2 = pd.DataFrame({'int': [4, 5, 6], 'str': ['d', 'e', 'f']}, columns=['str', 'int'], index=[2, 4, 6])
        self._run_test(lambda df1, df2: df1.append(df2, sort=True), df1, df2)
        self._run_test(lambda df1, df2: df1.append(df2, sort=False), df1, df2)
        self._run_test(lambda df1, df2: df2.append(df1, sort=True), df1, df2)
        self._run_test(lambda df1, df2: df2.append(df1, sort=False), df1, df2)

    def test_smallest_largest(self):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [2, 3, 5, 7]})
        self._run_test(lambda df: df.nlargest(1, 'A', keep='all'), df)
        self._run_test(lambda df: df.nsmallest(3, 'A', keep='all'), df)
        self._run_test(lambda df: df.nlargest(3, ['A', 'B'], keep='all'), df)

    def test_series_cov_corr(self):
        if False:
            for i in range(10):
                print('nop')
        for s in [pd.Series([1, 2, 3]), pd.Series(range(100)), pd.Series([x ** 3 for x in range(-50, 50)])]:
            self._run_test(lambda s: s.std(), s)
            self._run_test(lambda s: s.var(), s)
            self._run_test(lambda s: s.corr(s), s)
            self._run_test(lambda s: s.corr(s + 1), s)
            self._run_test(lambda s: s.corr(s * s), s)
            self._run_test(lambda s: s.cov(s * s), s)
            self._run_test(lambda s: s.skew(), s)
            self._run_test(lambda s: s.kurtosis(), s)
            self._run_test(lambda s: s.kurt(), s)

    def test_dataframe_cov_corr(self):
        if False:
            return 10
        df = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
        df.loc[df.index[:5], 'a'] = np.nan
        df.loc[df.index[5:10], 'b'] = np.nan
        self._run_test(lambda df: df.corr(), df)
        self._run_test(lambda df: df.cov(), df)
        self._run_test(lambda df: df.corr(min_periods=12), df)
        self._run_test(lambda df: df.cov(min_periods=12), df)
        self._run_test(lambda df: df.corrwith(df.a), df)
        self._run_test(lambda df: df[['a', 'b']].corrwith(df[['b', 'c']]), df)
        df2 = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
        self._run_test(lambda df, df2: df.corrwith(df2, axis=1), df, df2, check_proxy=False)

    def test_corrwith_bad_axis(self):
        if False:
            return 10
        df = pd.DataFrame({'a': range(3), 'b': range(3, 6), 'c': range(6, 9)})
        self._run_error_test(lambda df: df.corrwith(df.a, axis=2), df)
        self._run_error_test(lambda df: df.corrwith(df, axis=5), df)

    @unittest.skipIf(PD_VERSION < (1, 2), 'na_action added in pandas 1.2.0')
    @pytest.mark.filterwarnings('ignore:The default of observed=False is deprecated:FutureWarning')
    def test_applymap_na_action(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame([[pd.NA, 2.12], [3.356, 4.567]])
        self._run_test(lambda df: df.applymap(lambda x: len(str(x)), na_action='ignore'), df, check_proxy=False)

    @unittest.skipIf(PD_VERSION < (2, 1), 'map added in 2.1.0')
    def test_map_na_action(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame([[pd.NA, 2.12], [3.356, 4.567]])
        self._run_test(lambda df: df.map(lambda x: len(str(x)), na_action='ignore'), df, check_proxy=False)

    def test_dataframe_eval_query(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
        self._run_test(lambda df: df.eval('foo = a + b - c'), df)
        self._run_test(lambda df: df.query('a > b + c'), df)
        self._run_inplace_test(lambda df: df.eval('foo = a + b - c'), df)
        deferred_df = frame_base.DeferredFrame.wrap(expressions.ConstantExpression(df, df[0:0]))
        self.assertRaises(NotImplementedError, lambda : deferred_df.eval('foo = a + @b - c'))
        self.assertRaises(NotImplementedError, lambda : deferred_df.query('a > @b + c'))

    def test_index_name_assignment(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2]})
        df = df.set_index(['a', 'b'], drop=False)

        def change_index_names(df):
            if False:
                for i in range(10):
                    print('nop')
            df.index.names = ['A', None]
        self._run_inplace_test(change_index_names, df)

    def test_quantile(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]), columns=['a', 'b'])
        self._run_test(lambda df: df.quantile(0.1, axis='columns'), df, check_proxy=False)
        self._run_test(lambda df: df.quantile(0.1, axis='columns'), df, check_proxy=False)
        with self.assertRaisesRegex(frame_base.WontImplementError, "df\\.quantile\\(q=0\\.1, axis='columns'\\)"):
            self._run_test(lambda df: df.quantile([0.1, 0.5], axis='columns'), df)

    def test_dataframe_melt(self):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'}, 'B': {0: 1, 1: 3, 2: 5}, 'C': {0: 2, 1: 4, 2: 6}})
        self._run_test(lambda df: df.melt(id_vars=['A'], value_vars=['B'], ignore_index=False), df)
        self._run_test(lambda df: df.melt(id_vars=['A'], value_vars=['B', 'C'], ignore_index=False), df)
        self._run_test(lambda df: df.melt(id_vars=['A'], value_vars=['B'], var_name='myVarname', value_name='myValname', ignore_index=False), df)
        self._run_test(lambda df: df.melt(id_vars=['A'], value_vars=['B', 'C'], ignore_index=False), df)
        df.columns = [list('ABC'), list('DEF')]
        self._run_test(lambda df: df.melt(col_level=0, id_vars=['A'], value_vars=['B'], ignore_index=False), df)
        self._run_test(lambda df: df.melt(id_vars=[('A', 'D')], value_vars=[('B', 'E')], ignore_index=False), df)

    def test_fillna_columns(self):
        if False:
            return 10
        df = pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5], [np.nan, 3, np.nan, 4], [3, np.nan, np.nan, 4]], columns=list('ABCD'))
        self._run_test(lambda df: df.fillna(method='ffill', axis='columns'), df)
        self._run_test(lambda df: df.fillna(method='ffill', axis='columns', limit=1), df)
        self._run_test(lambda df: df.fillna(method='bfill', axis='columns', limit=1), df)

    def test_dataframe_fillna_dataframe_as_value(self):
        if False:
            return 10
        df = pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5], [np.nan, 3, np.nan, 4]], columns=list('ABCD'))
        df2 = pd.DataFrame(np.zeros((4, 4)), columns=list('ABCE'))
        self._run_test(lambda df, df2: df.fillna(df2), df, df2)

    def test_dataframe_fillna_series_as_value(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5], [np.nan, 3, np.nan, 4]], columns=list('ABCD'))
        s = pd.Series(range(4), index=list('ABCE'))
        self._run_test(lambda df, s: df.fillna(s), df, s)

    def test_series_fillna_series_as_value(self):
        if False:
            print('Hello World!')
        df = pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5], [np.nan, 3, np.nan, 4]], columns=list('ABCD'))
        df2 = pd.DataFrame(np.zeros((4, 4)), columns=list('ABCE'))
        self._run_test(lambda df, df2: df.A.fillna(df2.A), df, df2)

    @unittest.skipIf(PD_VERSION >= (2, 0), 'append removed in Pandas 2.0')
    def test_append_verify_integrity(self):
        if False:
            for i in range(10):
                print('nop')
        df1 = pd.DataFrame({'A': range(10), 'B': range(10)}, index=range(10))
        df2 = pd.DataFrame({'A': range(10), 'B': range(10)}, index=range(9, 19))
        self._run_error_test(lambda s1, s2: s1.append(s2, verify_integrity=True), df1['A'], df2['A'], construction_time=False)
        self._run_error_test(lambda df1, df2: df1.append(df2, verify_integrity=True), df1, df2, construction_time=False)

    def test_categorical_groupby(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'A': np.arange(6), 'B': list('aabbca')})
        df['B'] = df['B'].astype(pd.CategoricalDtype(list('cab')))
        df = df.set_index('B')
        self._run_test(lambda df: df.groupby(level=0, observed=False).sum(), df, nonparallel=True)
        self._run_test(lambda df: df.groupby(level=0, observed=False).mean(), df, nonparallel=True)

    def test_astype_categorical(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'A': np.arange(6), 'B': list('aabbca')})
        categorical_dtype = pd.CategoricalDtype(df.B.unique())
        self._run_test(lambda df: df.B.astype(categorical_dtype), df)

    @unittest.skipIf(PD_VERSION < (1, 2), 'DataFrame.unstack not supported in pandas <1.2.x')
    def test_astype_categorical_with_unstack(self):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'index1': ['one', 'one', 'two', 'two'], 'index2': ['a', 'b', 'a', 'b'], 'data': np.arange(1.0, 5.0)})

        def with_categorical_index(df):
            if False:
                while True:
                    i = 10
            df.index1 = df.index1.astype(pd.CategoricalDtype(['one', 'two']))
            df.index2 = df.index2.astype(pd.CategoricalDtype(['a', 'b']))
            df.set_index(['index1', 'index2'], drop=True)
            return df
        self._run_test(lambda df: with_categorical_index(df).unstack(level=-1), df, check_proxy=False)

    def test_dataframe_sum_nonnumeric_raises(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(frame_base.WontImplementError, 'numeric_only'):
            self._run_test(lambda df: df.sum(), GROUPBY_DF)
        self._run_test(lambda df: df.sum(numeric_only=True), GROUPBY_DF)
        self._run_test(lambda df: df[['foo', 'bar']].sum(), GROUPBY_DF)

    def test_insert(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        self._run_inplace_test(lambda df: df.insert(1, 'C', df.A * 2), df)
        self._run_inplace_test(lambda df: df.insert(0, 'foo', pd.Series([8], index=[1])), df, check_proxy=False)
        self._run_inplace_test(lambda df: df.insert(2, 'bar', value='q'), df)

    def test_insert_does_not_support_list_value(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        with self.assertRaisesRegex(frame_base.WontImplementError, 'insert\\(value=list\\)'):
            self._run_inplace_test(lambda df: df.insert(1, 'C', [7, 8, 9]), df)

    def test_drop_duplicates(self):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'], 'style': ['cup', 'cup', 'cup', 'pack', 'pack'], 'rating': [4, 4, 3.5, 15, 5]})
        self._run_test(lambda df: df.drop_duplicates(keep=False), df)
        self._run_test(lambda df: df.drop_duplicates(subset=['brand'], keep=False), df)
        self._run_test(lambda df: df.drop_duplicates(subset=['brand', 'style'], keep=False), df)

    @parameterized.expand([(lambda base: base.from_dict({'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}),), (lambda base: base.from_dict({'row_1': [3, 2, 1, 0], 'row_2': ['a', 'b', 'c', 'd']}, orient='index'),), (lambda base: base.from_records(np.array([(3, 'a'), (2, 'b'), (1, 'c'), (0, 'd')], dtype=[('col_1', 'i4'), ('col_2', 'U1')])),)])
    def test_create_methods(self, func):
        if False:
            for i in range(10):
                print('nop')
        expected = func(pd.DataFrame)
        deferred_df = func(frames.DeferredDataFrame)
        actual = expressions.Session({}).evaluate(deferred_df._expr)
        pd.testing.assert_frame_equal(actual, expected)

    def test_replace(self):
        if False:
            return 10
        df = pd.DataFrame({'A': ['bat', 'foo', 'bait'], 'B': ['abc', 'bar', 'xyz']})
        self._run_test(lambda df: df.replace(regex={'^ba.$': 'new', 'foo': 'xyz'}, method=None), df)

    def test_sample_columns(self):
        if False:
            return 10
        df = pd.DataFrame({'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'], 'style': ['cup', 'cup', 'cup', 'pack', 'pack'], 'rating': [4, 4, 3.5, 15, 5]})
        self._run_test(lambda df: df.sample(axis=1, n=2, random_state=1), df)
        self._run_error_test(lambda df: df.sample(axis=1, n=10, random_state=2), df)
        self._run_test(lambda df: df.sample(axis=1, n=10, random_state=3, replace=True), df)

    def test_cat(self):
        if False:
            return 10
        s = pd.Series(list('abbccc')).astype('category')
        self._run_test(lambda s: s.cat.rename_categories(list('cba')), s)
        self._run_test(lambda s: s.cat.reorder_categories(list('cba')), s)
        self._run_test(lambda s: s.cat.add_categories(['d', 'e']), s)
        self._run_test(lambda s: s.cat.remove_categories(['a', 'c']), s)
        self._run_test(lambda s: s.cat.set_categories(list('abcde')), s)
        self._run_test(lambda s: s.cat.as_ordered(), s)
        self._run_test(lambda s: s.cat.as_unordered(), s)
        self._run_test(lambda s: s.cat.codes, s)

    @parameterized.expand(frames.ELEMENTWISE_DATETIME_PROPERTIES)
    def test_dt_property(self, prop_name):
        if False:
            for i in range(10):
                print('nop')
        s = pd.Series(pd.date_range('1/1/2000', periods=100, freq='m') + pd.timedelta_range(start='0 days', end='70 days', periods=100))
        self._run_test(lambda s: getattr(s.dt, prop_name), s)

    @parameterized.expand([('month_name', {}), ('day_name', {}), ('normalize', {}), ('strftime', {'date_format': '%B %d, %Y, %r'}), ('tz_convert', {'tz': 'Europe/Berlin'})])
    def test_dt_method(self, op, kwargs):
        if False:
            for i in range(10):
                print('nop')
        s = pd.Series(pd.date_range('1/1/2000', periods=100, freq='m', tz='America/Los_Angeles') + pd.timedelta_range(start='0 days', end='70 days', periods=100))
        self._run_test(lambda s: getattr(s.dt, op)(**kwargs), s)

    def test_dt_tz_localize_ambiguous_series(self):
        if False:
            return 10
        s = pd.to_datetime(pd.Series(['2018-10-28 01:20:00', '2018-10-28 02:36:00', '2018-10-28 03:46:00']))
        ambiguous = pd.Series([True, True, False], index=s.index)
        self._run_test(lambda s, ambiguous: s.dt.tz_localize('CET', ambiguous=ambiguous), s, ambiguous)

    def test_dt_tz_localize_nonexistent(self):
        if False:
            for i in range(10):
                print('nop')
        s = pd.to_datetime(pd.Series(['2015-03-29 02:30:00', '2015-03-29 03:30:00']))
        self._run_test(lambda s: s.dt.tz_localize('Europe/Warsaw', ambiguous='NaT', nonexistent='shift_forward'), s)
        self._run_test(lambda s: s.dt.tz_localize('Europe/Warsaw', ambiguous='NaT', nonexistent='shift_backward'), s)
        self._run_test(lambda s: s.dt.tz_localize('Europe/Warsaw', ambiguous='NaT', nonexistent=pd.Timedelta('1H')), s)

    def test_compare_series(self):
        if False:
            print('Hello World!')
        s1 = pd.Series(['a', 'b', 'c', 'd', 'e'])
        s2 = pd.Series(['a', 'a', 'c', 'b', 'e'])
        self._run_test(lambda s1, s2: s1.compare(s2), s1, s2)
        self._run_test(lambda s1, s2: s1.compare(s2, align_axis=0), s1, s2)
        self._run_test(lambda s1, s2: s1.compare(s2, keep_shape=True), s1, s2)
        self._run_test(lambda s1, s2: s1.compare(s2, keep_shape=True, keep_equal=True), s1, s2)

    def test_compare_dataframe(self):
        if False:
            i = 10
            return i + 15
        df1 = pd.DataFrame({'col1': ['a', 'a', 'b', 'b', 'a'], 'col2': [1.0, 2.0, 3.0, np.nan, 5.0], 'col3': [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=['col1', 'col2', 'col3'])
        df2 = df1.copy()
        df2.loc[0, 'col1'] = 'c'
        df2.loc[2, 'col3'] = 4.0
        with self.assertRaisesRegex(frame_base.WontImplementError, 'compare\\(align_axis\\=1, keep_shape\\=False\\) is not allowed'):
            self._run_test(lambda df1, df2: df1.compare(df2), df1, df2)
        self._run_test(lambda df1, df2: df1.compare(df2, align_axis=0), df1, df2, check_proxy=False)
        self._run_test(lambda df1, df2: df1.compare(df2, keep_shape=True), df1, df2)
        self._run_test(lambda df1, df2: df1.compare(df2, align_axis=0, keep_shape=True), df1, df2)
        self._run_test(lambda df1, df2: df1.compare(df2, keep_shape=True, keep_equal=True), df1, df2)
        self._run_test(lambda df1, df2: df1.compare(df2, align_axis=0, keep_shape=True, keep_equal=True), df1, df2)

    def test_idxmin(self):
        if False:
            return 10
        df = pd.DataFrame({'consumption': [10.51, 103.11, 55.48], 'co2_emissions': [37.2, 19.66, 1712]}, index=['Pork', 'Wheat Products', 'Beef'])
        df2 = df.copy()
        df2.loc['Pork', 'co2_emissions'] = None
        df2.loc['Wheat Products', 'consumption'] = None
        df2.loc['Beef', 'co2_emissions'] = None
        df3 = pd.DataFrame({'consumption': [1.1, 2.2, 3.3], 'co2_emissions': [3.3, 2.2, 1.1]}, index=[0, 1, 2])
        s = pd.Series(data=[4, 3, None, 1], index=['A', 'B', 'C', 'D'])
        s2 = pd.Series(data=[1, 2, 3], index=[1, 2, 3])
        self._run_test(lambda df: df.idxmin(), df)
        self._run_test(lambda df: df.idxmin(skipna=False), df)
        self._run_test(lambda df: df.idxmin(axis=1), df)
        self._run_test(lambda df: df.idxmin(axis=1, skipna=False), df)
        self._run_test(lambda df2: df2.idxmin(), df2)
        self._run_test(lambda df2: df2.idxmin(axis=1), df2)
        self._run_test(lambda df2: df2.idxmin(skipna=False), df2, check_proxy=False)
        self._run_test(lambda df2: df2.idxmin(axis=1, skipna=False), df2, check_proxy=False)
        self._run_test(lambda df3: df3.idxmin(), df3)
        self._run_test(lambda df3: df3.idxmin(axis=1), df3)
        self._run_test(lambda df3: df3.idxmin(skipna=False), df3)
        self._run_test(lambda df3: df3.idxmin(axis=1, skipna=False), df3)
        self._run_test(lambda s: s.idxmin(), s)
        self._run_test(lambda s: s.idxmin(skipna=False), s, check_proxy=False)
        self._run_test(lambda s2: s2.idxmin(), s2)
        self._run_test(lambda s2: s2.idxmin(skipna=False), s2)

    def test_idxmax(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'consumption': [10.51, 103.11, 55.48], 'co2_emissions': [37.2, 19.66, 1712]}, index=['Pork', 'Wheat Products', 'Beef'])
        df2 = df.copy()
        df2.loc['Pork', 'co2_emissions'] = None
        df2.loc['Wheat Products', 'consumption'] = None
        df2.loc['Beef', 'co2_emissions'] = None
        df3 = pd.DataFrame({'consumption': [1.1, 2.2, 3.3], 'co2_emissions': [3.3, 2.2, 1.1]}, index=[0, 1, 2])
        s = pd.Series(data=[1, None, 4, 1], index=['A', 'B', 'C', 'D'])
        s2 = pd.Series(data=[1, 2, 3], index=[1, 2, 3])
        self._run_test(lambda df: df.idxmax(), df)
        self._run_test(lambda df: df.idxmax(skipna=False), df)
        self._run_test(lambda df: df.idxmax(axis=1), df)
        self._run_test(lambda df: df.idxmax(axis=1, skipna=False), df)
        self._run_test(lambda df2: df2.idxmax(), df2)
        self._run_test(lambda df2: df2.idxmax(axis=1), df2)
        self._run_test(lambda df2: df2.idxmax(axis=1, skipna=False), df2, check_proxy=False)
        self._run_test(lambda df2: df2.idxmax(skipna=False), df2, check_proxy=False)
        self._run_test(lambda df3: df3.idxmax(), df3)
        self._run_test(lambda df3: df3.idxmax(axis=1), df3)
        self._run_test(lambda df3: df3.idxmax(skipna=False), df3)
        self._run_test(lambda df3: df3.idxmax(axis=1, skipna=False), df3)
        self._run_test(lambda s: s.idxmax(), s)
        self._run_test(lambda s: s.idxmax(skipna=False), s, check_proxy=False)
        self._run_test(lambda s2: s2.idxmax(), s2)
        self._run_test(lambda s2: s2.idxmax(skipna=False), s2)

    def test_pipe(self):
        if False:
            for i in range(10):
                print('nop')

        def df_times(df, column, times):
            if False:
                while True:
                    i = 10
            df[column] = df[column] * times
            return df

        def df_times_shuffled(column, times, df):
            if False:
                i = 10
                return i + 15
            return df_times(df, column, times)

        def s_times(s, times):
            if False:
                while True:
                    i = 10
            return s * times

        def s_times_shuffled(times, s):
            if False:
                return 10
            return s_times(s, times)
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=[0, 1, 2])
        s = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])
        self._run_inplace_test(lambda df: df.pipe(df_times, 'A', 2), df)
        self._run_inplace_test(lambda df: df.pipe((df_times_shuffled, 'df'), 'A', 2), df)
        self._run_test(lambda s: s.pipe(s_times, 2), s)
        self._run_test(lambda s: s.pipe((s_times_shuffled, 's'), 2), s)

    def test_unstack_pandas_series_not_multiindex(self):
        if False:
            i = 10
            return i + 15
        s = pd.Series([1, 2, 3, 4], index=['one', 'two', 'three', 'four'])
        with self.assertRaises((AttributeError, ValueError)):
            self._run_test(lambda s: s.unstack(), s)

    def test_unstack_non_categorical_index(self):
        if False:
            i = 10
            return i + 15
        index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'), ('two', 'a'), ('two', 'b')])
        index = index.set_levels(index.levels[0].astype(pd.CategoricalDtype(['one', 'two'])), level=0)
        s = pd.Series(np.arange(1.0, 5.0), index=index)
        with self.assertRaisesRegex(frame_base.WontImplementError, 'unstack\\(\\) is only supported on DataFrames if'):
            self._run_test(lambda s: s.unstack(level=-1), s)

    def _unstack_get_categorical_index(self):
        if False:
            i = 10
            return i + 15
        index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'), ('two', 'a'), ('two', 'b')])
        index = index.set_levels(index.levels[0].astype(pd.CategoricalDtype(['one', 'two'])), level=0)
        index = index.set_levels(index.levels[1].astype(pd.CategoricalDtype(['a', 'b'])), level=1)
        return index

    def test_unstack_pandas_example1(self):
        if False:
            i = 10
            return i + 15
        index = self._unstack_get_categorical_index()
        s = pd.Series(np.arange(1.0, 5.0), index=index)
        self._run_test(lambda s: s.unstack(level=-1), s)

    def test_unstack_pandas_example2(self):
        if False:
            i = 10
            return i + 15
        index = self._unstack_get_categorical_index()
        s = pd.Series(np.arange(1.0, 5.0), index=index)
        self._run_test(lambda s: s.unstack(level=0), s)

    def test_unstack_pandas_example3(self):
        if False:
            return 10
        index = self._unstack_get_categorical_index()
        s = pd.Series(np.arange(1.0, 5.0), index=index)
        df = s.unstack(level=0)
        if PD_VERSION < (1, 2):
            with self.assertRaisesRegex(frame_base.WontImplementError, 'unstack\\(\\) is not supported when using pandas < 1.2.0'):
                self._run_test(lambda df: df.unstack(), df)
        else:
            self._run_test(lambda df: df.unstack(), df)

    @unittest.skipIf(PD_VERSION < (1, 4), 'Cannot set dtype of index to boolean for pandas<1.4')
    def test_unstack_bool(self):
        if False:
            for i in range(10):
                print('nop')
        index = pd.MultiIndex.from_tuples([(True, 'a'), (True, 'b'), (False, 'a'), (False, 'b')])
        index = index.set_levels(index.levels[0].astype('boolean'), level=0)
        index = index.set_levels(index.levels[1].astype(pd.CategoricalDtype(['a', 'b'])), level=1)
        s = pd.Series(np.arange(1.0, 5.0), index=index)
        self._run_test(lambda s: s.unstack(level=0), s)

    def test_unstack_series_multiple_index_levels(self):
        if False:
            i = 10
            return i + 15
        tuples = list(zip(*[['bar', 'bar', 'bar', 'bar', 'baz', 'baz', 'baz', 'baz'], ['one', 'one', 'two', 'two', 'one', 'one', 'two', 'two'], ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']]))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second', 'third'])
        index = index.set_levels(index.levels[0].astype(pd.CategoricalDtype(['bar', 'baz'])), level=0)
        index = index.set_levels(index.levels[1].astype(pd.CategoricalDtype(['one', 'two'])), level=1)
        index = index.set_levels(index.levels[2].astype(pd.CategoricalDtype(['A', 'B'])), level=2)
        df = pd.Series(np.random.randn(8), index=index)
        self._run_test(lambda df: df.unstack(level=['first', 'third']), df)

    def test_unstack_series_multiple_index_and_column_levels(self):
        if False:
            for i in range(10):
                print('nop')
        columns = pd.MultiIndex.from_tuples([('A', 'cat', 'long'), ('B', 'cat', 'long'), ('A', 'dog', 'short'), ('B', 'dog', 'short')], names=['exp', 'animal', 'hair_length'])
        index = pd.MultiIndex.from_product([['one', 'two'], ['a', 'b'], ['bar', 'baz']], names=['first', 'second', 'third'])
        index = index.set_levels(index.levels[0].astype(pd.CategoricalDtype(['one', 'two'])), level=0)
        index = index.set_levels(index.levels[1].astype(pd.CategoricalDtype(['a', 'b'])), level=1)
        index = index.set_levels(index.levels[2].astype(pd.CategoricalDtype(['bar', 'baz'])), level=2)
        df = pd.DataFrame(np.random.randn(8, 4), index=index, columns=columns)
        df = df.stack(level=['animal', 'hair_length'])
        self._run_test(lambda df: df.unstack(level=['second', 'third']), df)
        self._run_test(lambda df: df.unstack(level=['second']), df)

    def test_pivot_non_categorical(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'], 'bar': ['A', 'B', 'C', 'A', 'B', 'C'], 'baz': [1, 2, 3, 4, 5, 6], 'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
        with self.assertRaisesRegex(frame_base.WontImplementError, 'pivot\\(\\) of non-categorical type is not supported'):
            self._run_test(lambda df: df.pivot(index='foo', columns='bar', values='baz'), df)

    def test_pivot_pandas_example1(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'], 'bar': ['A', 'B', 'C', 'A', 'B', 'C'], 'baz': [1, 2, 3, 4, 5, 6], 'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
        df['bar'] = df['bar'].astype(pd.CategoricalDtype(categories=['A', 'B', 'C']))
        self._run_test(lambda df: df.pivot(index='foo', columns='bar', values='baz'), df)
        self._run_test(lambda df: df.pivot(index=['foo'], columns='bar', values='baz'), df)

    def test_pivot_pandas_example3(self):
        if False:
            return 10
        df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'], 'bar': ['A', 'B', 'C', 'A', 'B', 'C'], 'baz': [1, 2, 3, 4, 5, 6], 'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
        df['bar'] = df['bar'].astype(pd.CategoricalDtype(categories=['A', 'B', 'C']))
        self._run_test(lambda df: df.pivot(index='foo', columns='bar', values=['baz', 'zoo']), df)
        self._run_test(lambda df: df.pivot(index='foo', columns=['bar'], values=['baz', 'zoo']), df)

    def test_pivot_pandas_example4(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'lev1': [1, 1, 1, 2, 2, 2], 'lev2': [1, 1, 2, 1, 1, 2], 'lev3': [1, 2, 1, 2, 1, 2], 'lev4': [1, 2, 3, 4, 5, 6], 'values': [0, 1, 2, 3, 4, 5]})
        df['lev2'] = df['lev2'].astype(pd.CategoricalDtype(categories=[1, 2]))
        df['lev3'] = df['lev3'].astype(pd.CategoricalDtype(categories=[1, 2]))
        df['values'] = df['values'].astype('Int64')
        self._run_test(lambda df: df.pivot(index='lev1', columns=['lev2', 'lev3'], values='values'), df)

    def test_pivot_pandas_example5(self):
        if False:
            return 10
        df = pd.DataFrame({'lev1': [1, 1, 1, 2, 2, 2], 'lev2': [1, 1, 2, 1, 1, 2], 'lev3': [1, 2, 1, 2, 1, 2], 'lev4': [1, 2, 3, 4, 5, 6], 'values': [0, 1, 2, 3, 4, 5]})
        df['lev3'] = df['lev3'].astype(pd.CategoricalDtype(categories=[1, 2]))
        df['values'] = df['values'].astype('Int64')
        if PD_VERSION < (1, 4):
            with self.assertRaisesRegex(frame_base.WontImplementError, 'pivot\\(\\) is not supported when pandas<1.4 and index is a Multi'):
                self._run_test(lambda df: df.pivot(index=['lev1', 'lev2'], columns=['lev3'], values='values'), df)
        else:
            self._run_test(lambda df: df.pivot(index=['lev1', 'lev2'], columns=['lev3'], values='values'), df)

    def test_pivot_pandas_example6(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'foo': ['one', 'one', 'two', 'two'], 'bar': ['A', 'A', 'B', 'C'], 'baz': [1, 2, 3, 4]})
        df['bar'] = df['bar'].astype(pd.CategoricalDtype(categories=['A', 'B', 'C']))
        self._run_error_test(lambda df: df.pivot(index='foo', columns='bar', values='baz'), df, construction_time=False)

    def test_pivot_no_index_provided_on_single_level_index(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'lev1': [1, 1, 1, 2, 2, 2], 'lev2': [1, 1, 2, 1, 1, 2], 'lev3': [1, 2, 1, 2, 1, 2], 'lev4': [1, 2, 3, 4, 5, 6], 'values': [0, 1, 2, 3, 4, 5]})
        df['lev2'] = df['lev2'].astype(pd.CategoricalDtype(categories=[1, 2]))
        df['lev3'] = df['lev3'].astype(pd.CategoricalDtype(categories=[1, 2]))
        df['values'] = df['values'].astype('Int64')
        self._run_test(lambda df: df.pivot(columns=['lev2', 'lev3'], values='values'), df)

    def test_pivot_no_index_provided_on_multiindex(self):
        if False:
            print('Hello World!')
        tuples = list(zip(*[['bar', 'bar', 'bar', 'baz', 'baz', 'baz'], ['one', 'two', 'three', 'one', 'two', 'three']]))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
        df = pd.DataFrame({'lev1': [1, 1, 1, 2, 2, 2], 'lev2': [1, 1, 2, 1, 1, 2], 'lev3': [1, 2, 1, 2, 1, 2], 'lev4': [1, 2, 3, 4, 5, 6], 'values': [0, 1, 2, 3, 4, 5]}, index=index)
        df['lev2'] = df['lev2'].astype(pd.CategoricalDtype(categories=[1, 2]))
        df['lev3'] = df['lev3'].astype(pd.CategoricalDtype(categories=[1, 2]))
        df['values'] = df['values'].astype('float64')
        df['lev1'] = df['lev1'].astype('int64')
        df['lev4'] = df['lev4'].astype('int64')
        if PD_VERSION < (1, 4):
            with self.assertRaisesRegex(frame_base.WontImplementError, 'pivot\\(\\) is not supported when pandas<1.4 and index is a Multi'):
                self._run_test(lambda df: df.pivot(columns=['lev2', 'lev3']), df)
        else:
            self._run_test(lambda df: df.pivot(columns=['lev2', 'lev3']), df, lenient_dtype_check=True)
ALL_GROUPING_AGGREGATIONS = sorted(set(frames.ALL_AGGREGATIONS) - set(('kurt', 'kurtosis')))
AGGREGATIONS_WHERE_NUMERIC_ONLY_DEFAULTS_TO_TRUE_IN_PANDAS_1 = set(frames.ALL_AGGREGATIONS) - set(('nunique', 'size', 'count', 'idxmin', 'idxmax', 'mode', 'rank', 'all', 'any', 'describe'))

def numeric_only_kwargs_for_pandas_2(agg_type: str) -> Dict[str, bool]:
    if False:
        return 10
    'Get proper arguments for numeric_only.\n\n  Behavior for numeric_only in these methods changed in Pandas 2 to default\n  to False instead of True, so explicitly make it True in Pandas 2.'
    if PD_VERSION >= (2, 0) and agg_type in AGGREGATIONS_WHERE_NUMERIC_ONLY_DEFAULTS_TO_TRUE_IN_PANDAS_1:
        return {'numeric_only': True}
    else:
        return {}

class GroupByTest(_AbstractFrameTest):
    """Tests for DataFrame/Series GroupBy operations."""

    @staticmethod
    def median_sum_fn(x):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            return (x.foo + x.bar).median()

    @parameterized.expand(ALL_GROUPING_AGGREGATIONS)
    def test_groupby_agg(self, agg_type):
        if False:
            print('Hello World!')
        if agg_type == 'describe' and PD_VERSION < (1, 2):
            self.skipTest('https://github.com/apache/beam/issues/20967: proxy generation of DataFrameGroupBy.describe fails in pandas < 1.2')
        kwargs = numeric_only_kwargs_for_pandas_2(agg_type)
        self._run_test(lambda df: df.groupby('group').agg(agg_type, **kwargs), GROUPBY_DF, check_proxy=False)

    @parameterized.expand(ALL_GROUPING_AGGREGATIONS)
    def test_groupby_with_filter(self, agg_type):
        if False:
            i = 10
            return i + 15
        if agg_type == 'describe' and PD_VERSION < (1, 2):
            self.skipTest('https://github.com/apache/beam/issues/20967: proxy generation of DataFrameGroupBy.describe fails in pandas < 1.2')
        kwargs = numeric_only_kwargs_for_pandas_2(agg_type)
        self._run_test(lambda df: getattr(df[df.foo > 30].groupby('group'), agg_type)(**kwargs), GROUPBY_DF, check_proxy=False)

    @parameterized.expand(ALL_GROUPING_AGGREGATIONS)
    def test_groupby(self, agg_type):
        if False:
            for i in range(10):
                print('nop')
        if agg_type == 'describe' and PD_VERSION < (1, 2):
            self.skipTest('https://github.com/apache/beam/issues/20967: proxy generation of DataFrameGroupBy.describe fails in pandas < 1.2')
        kwargs = numeric_only_kwargs_for_pandas_2(agg_type)
        self._run_test(lambda df: getattr(df.groupby('group'), agg_type)(**kwargs), GROUPBY_DF, check_proxy=False)

    @parameterized.expand(ALL_GROUPING_AGGREGATIONS)
    def test_groupby_series(self, agg_type):
        if False:
            return 10
        if agg_type == 'describe' and PD_VERSION < (1, 2):
            self.skipTest('https://github.com/apache/beam/issues/20967: proxy generation of DataFrameGroupBy.describe fails in pandas < 1.2')
        kwargs = numeric_only_kwargs_for_pandas_2(agg_type)
        self._run_test(lambda df: getattr(df[df.foo > 40].groupby(df.group), agg_type)(**kwargs), GROUPBY_DF, check_proxy=False)

    def test_groupby_user_guide(self):
        if False:
            print('Hello World!')
        arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
        index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
        df = pd.DataFrame({'A': [1, 1, 1, 1, 2, 2, 3, 3], 'B': np.arange(8)}, index=index)
        self._run_test(lambda df: df.groupby(['second', 'A']).sum(), df)

    @parameterized.expand(ALL_GROUPING_AGGREGATIONS)
    def test_groupby_project_series(self, agg_type):
        if False:
            i = 10
            return i + 15
        df = GROUPBY_DF
        if agg_type == 'describe':
            self.skipTest('https://github.com/apache/beam/issues/20967: proxy generation of SeriesGroupBy.describe fails')
        if agg_type in ('corr', 'cov'):
            self.skipTest('https://github.com/apache/beam/issues/20895: SeriesGroupBy.{corr, cov} do not raise the expected error.')
        kwargs = numeric_only_kwargs_for_pandas_2(agg_type)
        self._run_test(lambda df: getattr(df.groupby('group').foo, agg_type)(**kwargs), df)
        self._run_test(lambda df: getattr(df.groupby('group').bar, agg_type)(**kwargs), df)
        self._run_test(lambda df: getattr(df.groupby('group')['foo'], agg_type)(**kwargs), df)
        self._run_test(lambda df: getattr(df.groupby('group')['bar'], agg_type)(**kwargs), df)

    @parameterized.expand(ALL_GROUPING_AGGREGATIONS)
    def test_groupby_project_dataframe(self, agg_type):
        if False:
            for i in range(10):
                print('nop')
        if agg_type == 'describe' and PD_VERSION < (1, 2):
            self.skipTest('https://github.com/apache/beam/issues/20967: proxy generation of DataFrameGroupBy.describe fails in pandas < 1.2')
        kwargs = numeric_only_kwargs_for_pandas_2(agg_type)
        self._run_test(lambda df: getattr(df.groupby('group')[['bar', 'baz']], agg_type)(**kwargs), GROUPBY_DF, check_proxy=False)

    def test_groupby_errors_bad_projection(self):
        if False:
            while True:
                i = 10
        df = GROUPBY_DF
        self._run_error_test(lambda df: df.groupby('group')[['bar', 'baz']].bar.median(), df)
        self._run_error_test(lambda df: df.groupby('group')[['bad']].median(), df)
        self._run_error_test(lambda df: df.groupby('group').bad.median(), df)
        self._run_error_test(lambda df: df.groupby('group')[['bar', 'baz']].bar.sum(), df)
        self._run_error_test(lambda df: df.groupby('group')[['bat']].sum(), df)
        self._run_error_test(lambda df: df.groupby('group').bat.sum(), df)

    def test_groupby_errors_non_existent_label(self):
        if False:
            while True:
                i = 10
        df = GROUPBY_DF
        self._run_error_test(lambda df: df.groupby(['really_bad', 'foo', 'bad']).foo.sum(), df)
        self._run_error_test(lambda df: df.groupby('bad').foo.sum(), df)

    def test_groupby_callable(self):
        if False:
            print('Hello World!')
        df = GROUPBY_DF
        kwargs = numeric_only_kwargs_for_pandas_2('sum')
        self._run_test(lambda df: df.groupby(lambda x: x % 2).foo.sum(**kwargs), df)
        kwargs = numeric_only_kwargs_for_pandas_2('median')
        self._run_test(lambda df: df.groupby(lambda x: x % 5).median(**kwargs), df)

    def test_groupby_apply(self):
        if False:
            while True:
                i = 10
        df = GROUPBY_DF
        describe = lambda df: df.describe()
        self._run_test(lambda df: df.groupby('group').foo.apply(describe), df)
        self._run_test(lambda df: df.groupby('group')[['foo', 'bar']].apply(describe), df)
        self._run_test(lambda df: df.groupby('group').apply(self.median_sum_fn), df)
        self._run_test(lambda df: df.set_index('group').foo.groupby(level=0).apply(describe), df)
        self._run_test(lambda df: df.groupby(level=0).apply(self.median_sum_fn), df)
        self._run_test(lambda df: df.groupby(lambda x: x % 3).apply(describe), df)
        self._run_test(lambda df: df.bar.groupby(lambda x: x % 3).apply(describe), df)
        self._run_test(lambda df: df.set_index(['str', 'group', 'bool']).groupby(level='group').apply(self.median_sum_fn), df)

    def test_groupby_apply_preserves_column_order(self):
        if False:
            i = 10
            return i + 15
        df = GROUPBY_DF
        self._run_test(lambda df: df[['foo', 'group', 'bar']].groupby('group', group_keys=False).apply(lambda x: x), df)

    def test_groupby_transform(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'Date': ['2015-05-08', '2015-05-07', '2015-05-06', '2015-05-05', '2015-05-08', '2015-05-07', '2015-05-06', '2015-05-05'], 'Data': [5, 8, 6, 1, 50, 100, 60, 120]})
        self._run_test(lambda df: df.groupby('Date')['Data'].transform(np.sum), df)
        self._run_test(lambda df: df.groupby('Date')['Data'].transform(lambda x: (x - x.mean()) / x.std()), df)

    def test_groupby_pipe(self):
        if False:
            i = 10
            return i + 15
        df = GROUPBY_DF
        kwargs = numeric_only_kwargs_for_pandas_2('sum')
        self._run_test(lambda df: df.groupby('group').pipe(lambda x: x.sum(**kwargs)), df)
        self._run_test(lambda df: df.groupby('group')['bool'].pipe(lambda x: x.any()), df)
        self._run_test(lambda df: df.groupby(['group', 'foo']).pipe((lambda a, x: x.sum(numeric_only=a), 'x'), False), df, check_proxy=False)

    def test_groupby_apply_modified_index(self):
        if False:
            while True:
                i = 10
        df = GROUPBY_DF
        self._run_test(lambda df: df.groupby('group').apply(lambda x: x[x.foo > x.foo.median()]), df)

    @unittest.skip('https://github.com/apache/beam/issues/20762')
    def test_groupby_aggregate_grouped_column(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'group': ['a' if i % 5 == 0 or i % 3 == 0 else 'b' for i in range(100)], 'foo': [None if i % 11 == 0 else i for i in range(100)], 'bar': [None if i % 7 == 0 else 99 - i for i in range(100)], 'baz': [None if i % 13 == 0 else i * 2 for i in range(100)]})
        self._run_test(lambda df: df.groupby('group').group.count(), df)
        self._run_test(lambda df: df.groupby('group')[['group', 'bar']].count(), df)
        self._run_test(lambda df: df.groupby('group')[['group', 'bar']].apply(lambda x: x.describe()), df)

    @parameterized.expand(((x,) for x in [0, [1], 3, [0, 3], [2, 1], ['foo', 0], [1, 'str'], [3, 0, 2, 1]]))
    def test_groupby_level_agg(self, level):
        if False:
            i = 10
            return i + 15
        df = GROUPBY_DF.set_index(['group', 'foo', 'bar', 'str'], drop=False)
        self._run_test(lambda df: df.groupby(level=level).bar.max(), df)
        self._run_test(lambda df: df.groupby(level=level).sum(numeric_only=True), df)
        self._run_test(lambda df: df.groupby(level=level).apply(self.median_sum_fn), df)

    @unittest.skipIf(PD_VERSION < (1, 1), 'drop_na added in pandas 1.1.0')
    def test_groupby_count_na(self):
        if False:
            return 10
        self._run_test(lambda df: df.groupby('foo', dropna=True).bar.count(), GROUPBY_DF)
        self._run_test(lambda df: df.groupby('foo', dropna=False).bar.count(), GROUPBY_DF)

    def test_groupby_sum_min_count(self):
        if False:
            return 10
        df = pd.DataFrame({'good': [1, 2, 3, np.nan], 'bad': [np.nan, np.nan, np.nan, 4], 'group': ['a', 'b', 'a', 'b']})
        self._run_test(lambda df: df.groupby('group').sum(min_count=2), df)

    @unittest.skipIf(PD_VERSION >= (2, 0), 'dtypes on groups is deprecated in Pandas 2.')
    def test_groupby_dtypes(self):
        if False:
            while True:
                i = 10
        self._run_test(lambda df: df.groupby('group').dtypes, GROUPBY_DF, check_proxy=False)
        self._run_test(lambda df: df.groupby(level=0).dtypes, GROUPBY_DF, check_proxy=False)

    @parameterized.expand(ALL_GROUPING_AGGREGATIONS)
    def test_dataframe_groupby_series(self, agg_type):
        if False:
            while True:
                i = 10
        if agg_type == 'describe' and PD_VERSION < (1, 2):
            self.skipTest('https://github.com/apache/beam/issues/20967: proxy generation of DataFrameGroupBy.describe fails in pandas < 1.2')

        def agg(df, group_by):
            if False:
                for i in range(10):
                    print('nop')
            kwargs = numeric_only_kwargs_for_pandas_2(agg_type)
            return df[df.foo > 40].groupby(group_by).agg(agg_type, **kwargs)
        self._run_test(lambda df: agg(df, df.group), GROUPBY_DF, check_proxy=False)
        self._run_test(lambda df: agg(df, df.foo % 3), GROUPBY_DF, check_proxy=False)

    @parameterized.expand(ALL_GROUPING_AGGREGATIONS)
    def test_series_groupby_series(self, agg_type):
        if False:
            while True:
                i = 10
        if agg_type == 'describe':
            self.skipTest('https://github.com/apache/beam/issues/20967: proxy generation of SeriesGroupBy.describe fails')
        if agg_type in ('corr', 'cov'):
            self.skipTest('https://github.com/apache/beam/issues/20895: SeriesGroupBy.{corr, cov} do not raise the expected error.')
        self._run_test(lambda df: df[df.foo < 40].bar.groupby(df.group).agg(agg_type), GROUPBY_DF)
        self._run_test(lambda df: df[df.foo < 40].bar.groupby(df.foo % 3).agg(agg_type), GROUPBY_DF)

    def test_groupby_series_apply(self):
        if False:
            for i in range(10):
                print('nop')
        df = GROUPBY_DF
        describe = lambda df: df.describe()
        self._run_test(lambda df: df.groupby(df.group).foo.apply(describe), df)
        self._run_test(lambda df: df.groupby(df.group)[['foo', 'bar']].apply(describe), df)
        self._run_test(lambda df: df.groupby(df.group).apply(self.median_sum_fn), df)

    def test_groupby_multiindex_keep_nans(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(NotImplementedError, 'https://github.com/apache/beam/issues/21014'):
            self._run_test(lambda df: df.groupby(['foo', 'bar'], dropna=False).sum(), GROUPBY_DF)
NONPARALLEL_METHODS = ['quantile', 'describe', 'median', 'sem']
if PD_VERSION < (2, 0):
    NONPARALLEL_METHODS.append('mad')

class AggregationTest(_AbstractFrameTest):
    """Tests for global aggregation methods on DataFrame/Series."""

    @parameterized.expand(sorted(set(frames.ALL_AGGREGATIONS) - set(['corr', 'cov'])))
    def test_series_agg(self, agg_method):
        if False:
            for i in range(10):
                print('nop')
        s = pd.Series(list(range(16)))
        nonparallel = agg_method in NONPARALLEL_METHODS
        check_proxy = agg_method not in ('max', 'min')
        self._run_test(lambda s: s.agg(agg_method), s, nonparallel=nonparallel, check_proxy=check_proxy)

    @parameterized.expand(sorted(set(frames.ALL_AGGREGATIONS) - set(['corr', 'cov', 'size'])))
    def test_series_agg_method(self, agg_method):
        if False:
            return 10
        s = pd.Series(list(range(16)))
        nonparallel = agg_method in NONPARALLEL_METHODS
        check_proxy = agg_method not in ('max', 'min')
        self._run_test(lambda s: getattr(s, agg_method)(), s, nonparallel=nonparallel, check_proxy=check_proxy)

    @parameterized.expand(frames.ALL_AGGREGATIONS)
    def test_dataframe_agg(self, agg_method):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, 3, 5, 7]})
        nonparallel = agg_method in NONPARALLEL_METHODS
        check_proxy = agg_method not in ('max', 'min')
        self._run_test(lambda df: df.agg(agg_method), df, nonparallel=nonparallel, check_proxy=check_proxy)

    @parameterized.expand(sorted(set(frames.ALL_AGGREGATIONS) - set(['size'])))
    def test_dataframe_agg_method(self, agg_method):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, 3, 5, 7]})
        nonparallel = agg_method in NONPARALLEL_METHODS
        check_proxy = agg_method not in ('max', 'min')
        self._run_test(lambda df: getattr(df, agg_method)(), df, nonparallel=nonparallel, check_proxy=check_proxy)

    def test_series_agg_modes(self):
        if False:
            print('Hello World!')
        s = pd.Series(list(range(16)))
        self._run_test(lambda s: s.agg('sum'), s)
        self._run_test(lambda s: s.agg(['sum']), s)
        self._run_test(lambda s: s.agg(['sum', 'mean']), s)
        self._run_test(lambda s: s.agg(['mean']), s)
        self._run_test(lambda s: s.agg('mean'), s)

    def test_dataframe_agg_modes(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, 3, 5, 7]})
        self._run_test(lambda df: df.agg('sum'), df)
        self._run_test(lambda df: df.agg(['sum', 'mean']), df)
        self._run_test(lambda df: df.agg({'A': 'sum', 'B': 'sum'}), df)
        self._run_test(lambda df: df.agg({'A': 'sum', 'B': 'mean'}), df)
        self._run_test(lambda df: df.agg({'A': ['sum', 'mean']}), df)
        self._run_test(lambda df: df.agg({'A': ['sum', 'mean'], 'B': 'min'}), df)

    @unittest.skipIf(PD_VERSION >= (2, 0), 'level argument removed in Pandas 2')
    def test_series_agg_level(self):
        if False:
            while True:
                i = 10
        self._run_test(lambda df: df.set_index(['group', 'foo']).bar.count(level=0), GROUPBY_DF)
        self._run_test(lambda df: df.set_index(['group', 'foo']).bar.max(level=0), GROUPBY_DF)
        self._run_test(lambda df: df.set_index(['group', 'foo']).bar.median(level=0), GROUPBY_DF)
        self._run_test(lambda df: df.set_index(['foo', 'group']).bar.count(level=1), GROUPBY_DF)
        self._run_test(lambda df: df.set_index(['group', 'foo']).bar.max(level=1), GROUPBY_DF)
        self._run_test(lambda df: df.set_index(['group', 'foo']).bar.max(level='foo'), GROUPBY_DF)
        self._run_test(lambda df: df.set_index(['group', 'foo']).bar.median(level=1), GROUPBY_DF)

    @unittest.skipIf(PD_VERSION >= (2, 0), 'level argument removed in Pandas 2')
    def test_dataframe_agg_level(self):
        if False:
            i = 10
            return i + 15
        self._run_test(lambda df: df.set_index(['group', 'foo']).count(level=0), GROUPBY_DF)
        self._run_test(lambda df: df.set_index(['group', 'foo']).max(level=0, numeric_only=False), GROUPBY_DF, check_proxy=False)
        self._run_test(lambda df: df.set_index(['group', 'foo']).sum(level=0, numeric_only=True), GROUPBY_DF)
        self._run_test(lambda df: df.set_index(['group', 'foo'])[['bar']].count(level=1), GROUPBY_DF)
        self._run_test(lambda df: df.set_index(['group', 'foo']).count(level=1), GROUPBY_DF)
        self._run_test(lambda df: df.set_index(['group', 'foo']).max(level=1, numeric_only=False), GROUPBY_DF, check_proxy=False)
        self._run_test(lambda df: df.set_index(['group', 'foo']).sum(level=1, numeric_only=True), GROUPBY_DF)
        self._run_test(lambda df: df.set_index(['group', 'foo']).median(level=0, numeric_only=True), GROUPBY_DF)
        self._run_test(lambda df: df.drop('str', axis=1).set_index(['foo', 'group']).median(level=1, numeric_only=True), GROUPBY_DF)

    @unittest.skipIf(PD_VERSION >= (2, 0), 'level argument removed in Pandas 2')
    def test_series_agg_multifunc_level(self):
        if False:
            i = 10
            return i + 15
        self._run_test(lambda df: df.set_index(['group', 'foo']).bar.agg(['min', 'max'], level=0), GROUPBY_DF)

    def test_series_mean_skipna(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'one': [i if i % 8 == 0 else np.nan for i in range(8)], 'two': [i if i % 4 == 0 else np.nan for i in range(8)], 'three': [i if i % 2 == 0 else np.nan for i in range(8)]})
        self._run_test(lambda df: df.one.mean(skipna=False), df)
        self._run_test(lambda df: df.two.mean(skipna=False), df)
        self._run_test(lambda df: df.three.mean(skipna=False), df)
        self._run_test(lambda df: df.one.mean(skipna=True), df)
        self._run_test(lambda df: df.two.mean(skipna=True), df)
        self._run_test(lambda df: df.three.mean(skipna=True), df)

    @unittest.skipIf(PD_VERSION >= (2, 0), 'level argument removed in Pandas 2')
    def test_dataframe_agg_multifunc_level(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_test(lambda df: df.set_index(['group', 'foo']).agg(['min', 'max'], level=0), GROUPBY_DF, check_proxy=False)

    @parameterized.expand([(True,), (False,)])
    @unittest.skipIf(PD_VERSION < (1, 2), 'pandas 1.1.0 produces different dtypes for these examples')
    def test_dataframe_agg_numeric_only(self, numeric_only):
        if False:
            print('Hello World!')
        self._run_test(lambda df: df.max(numeric_only=numeric_only), GROUPBY_DF, check_proxy=False)
        self._run_test(lambda df: df.min(numeric_only=numeric_only), GROUPBY_DF, check_proxy=False)

    @unittest.skip("pandas implementation doesn't respect numeric_only= with level= (https://github.com/pandas-dev/pandas/issues/40788)")
    def test_dataframe_agg_level_numeric_only(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_test(lambda df: df.set_index('foo').sum(level=0, numeric_only=True), GROUPBY_DF)
        self._run_test(lambda df: df.set_index('foo').max(level=0, numeric_only=True), GROUPBY_DF)
        self._run_test(lambda df: df.set_index('foo').mean(level=0, numeric_only=True), GROUPBY_DF)
        self._run_test(lambda df: df.set_index('foo').median(level=0, numeric_only=True), GROUPBY_DF)

    def test_dataframe_agg_bool_only(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'all': [True for i in range(10)], 'any': [i % 3 == 0 for i in range(10)], 'int': range(10)})
        self._run_test(lambda df: df.all(), df)
        self._run_test(lambda df: df.any(), df)
        self._run_test(lambda df: df.all(bool_only=True), df)
        self._run_test(lambda df: df.any(bool_only=True), df)

    @unittest.skip("pandas doesn't implement bool_only= with level= (https://github.com/pandas-dev/pandas/blob/v1.2.3/pandas/core/generic.py#L10573)")
    def test_dataframe_agg_level_bool_only(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'all': [True for i in range(10)], 'any': [i % 3 == 0 for i in range(10)], 'int': range(10)})
        self._run_test(lambda df: df.set_index('int', drop=False).all(level=0), df)
        self._run_test(lambda df: df.set_index('int', drop=False).any(level=0), df)
        self._run_test(lambda df: df.set_index('int', drop=False).all(level=0, bool_only=True), df)
        self._run_test(lambda df: df.set_index('int', drop=False).any(level=0, bool_only=True), df)

    def test_series_agg_np_size(self):
        if False:
            print('Hello World!')
        self._run_test(lambda df: df.set_index(['group', 'foo']).agg(np.size), GROUPBY_DF, check_proxy=False)

    def test_df_agg_invalid_kwarg_raises(self):
        if False:
            i = 10
            return i + 15
        self._run_error_test(lambda df: df.agg('mean', bool_only=True), GROUPBY_DF)
        self._run_error_test(lambda df: df.agg('any', numeric_only=True), GROUPBY_DF)
        self._run_error_test(lambda df: df.agg('median', min_count=3, numeric_only=True), GROUPBY_DF)

    def test_series_agg_method_invalid_kwarg_raises(self):
        if False:
            return 10
        self._run_error_test(lambda df: df.foo.median(min_count=3), GROUPBY_DF)
        self._run_error_test(lambda df: df.foo.agg('median', min_count=3), GROUPBY_DF)

    @unittest.skipIf(PD_VERSION < (1, 3), 'DataFrame.agg raises a different exception from the aggregation methods. Fixed in https://github.com/pandas-dev/pandas/pull/40543.')
    def test_df_agg_method_invalid_kwarg_raises(self):
        if False:
            return 10
        self._run_error_test(lambda df: df.mean(bool_only=True), GROUPBY_DF)
        self._run_error_test(lambda df: df.any(numeric_only=True), GROUPBY_DF)
        self._run_error_test(lambda df: df.median(min_count=3, numeric_only=True), GROUPBY_DF)

    @unittest.skipIf(PD_VERSION >= (2, 0), 'level argument removed in Pandas 2')
    def test_agg_min_count(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'good': [1, 2, 3, np.nan], 'bad': [np.nan, np.nan, np.nan, 4]}, index=['a', 'b', 'a', 'b'])
        self._run_test(lambda df: df.sum(level=0, min_count=2), df)
        self._run_test(lambda df: df.sum(min_count=3), df, nonparallel=True)
        self._run_test(lambda df: df.sum(min_count=1), df, nonparallel=True)
        self._run_test(lambda df: df.good.sum(min_count=2), df, nonparallel=True)
        self._run_test(lambda df: df.bad.sum(min_count=2), df, nonparallel=True)

    def test_series_agg_std(self):
        if False:
            for i in range(10):
                print('nop')
        s = pd.Series(range(10))
        self._run_test(lambda s: s.agg('std'), s)
        self._run_test(lambda s: s.agg('var'), s)
        self._run_test(lambda s: s.agg(['std', 'sum']), s)
        self._run_test(lambda s: s.agg(['var']), s)

    def test_std_all_na(self):
        if False:
            while True:
                i = 10
        s = pd.Series([np.nan] * 10)
        self._run_test(lambda s: s.agg('std'), s)
        self._run_test(lambda s: s.std(), s)

    def test_std_mostly_na_with_ddof(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'one': [i if i % 8 == 0 else np.nan for i in range(8)], 'two': [i if i % 4 == 0 else np.nan for i in range(8)], 'three': [i if i % 2 == 0 else np.nan for i in range(8)]}, index=pd.MultiIndex.from_arrays([list(range(8)), list(reversed(range(8)))], names=['forward', None]))
        self._run_test(lambda df: df.std(), df)
        self._run_test(lambda df: df.std(ddof=0), df)
        self._run_test(lambda df: df.std(ddof=2), df)
        self._run_test(lambda df: df.std(ddof=3), df)
        self._run_test(lambda df: df.std(ddof=4), df)

    def test_dataframe_std(self):
        if False:
            while True:
                i = 10
        self._run_test(lambda df: df.std(numeric_only=True), GROUPBY_DF)
        self._run_test(lambda df: df.var(numeric_only=True), GROUPBY_DF)

    def test_dataframe_mode(self):
        if False:
            while True:
                i = 10
        self._run_test(lambda df: df.mode(), GROUPBY_DF, nonparallel=True, check_proxy=False)
        self._run_test(lambda df: df.mode(numeric_only=True), GROUPBY_DF, nonparallel=True, check_proxy=False)
        self._run_test(lambda df: df.mode(dropna=True, numeric_only=True), GROUPBY_DF, nonparallel=True, check_proxy=False)

    def test_series_mode(self):
        if False:
            return 10
        self._run_test(lambda df: df.foo.mode(), GROUPBY_DF, nonparallel=True)
        self._run_test(lambda df: df.baz.mode(dropna=True), GROUPBY_DF, nonparallel=True)

class BeamSpecificTest(unittest.TestCase):
    """Tests for functionality that's specific to the Beam DataFrame API.

  These features don't exist in pandas so we must verify them independently."""

    def assert_frame_data_equivalent(self, actual, expected, check_column_subset=False, extra_col_value=0):
        if False:
            while True:
                i = 10
        'Verify that actual is the same as expected, ignoring the index and order\n    of the data.\n\n    Note: In order to perform non-deferred column operations in Beam, we have\n    to enumerate all possible categories of data, even if they are ultimately\n    unobserved. The default Pandas implementation on the other hand does not\n    produce unobserved columns. This means when conducting tests, we need to\n    account for the fact that the Beam result may be a superset of that of the\n    Pandas result.\n\n    If ``check_column_subset`` is `True`, we verify that all of the columns in\n    the Dataframe returned from the Pandas implementation is contained in the\n    Dataframe created from the Beam implementation.\n\n    We also check if all columns that exist in the Beam implementation but\n    not in the Pandas implementation are all equal to the ``extra_col_value``\n    to ensure that they were not erroneously populated.\n    '
        if check_column_subset:
            if isinstance(expected, pd.DataFrame):
                expected_cols = set(expected.columns)
                actual_cols = set(actual.columns)
                if not set(expected_cols).issubset(set(actual_cols)):
                    raise AssertionError(f'Expected columns:\n{expected.columns}\n is not asubset of {actual.columns}.')
                extra_columns = actual_cols - expected_cols
                if extra_columns:
                    actual_extra_only = actual[list(extra_columns)]
                    if np.isnan(extra_col_value):
                        extra_cols_all_match = actual_extra_only.isna().all().all()
                    else:
                        extra_cols_all_match = actual_extra_only.eq(extra_col_value).all().all()
                    if not extra_cols_all_match:
                        raise AssertionError(f'Extra columns:{extra_columns}\n should all be {extra_col_value}, but got \n{actual_extra_only}.')
                actual = actual[expected.columns]

        def sort_and_drop_index(df):
            if False:
                i = 10
                return i + 15
            if isinstance(df, pd.Series):
                df = df.sort_values()
            elif isinstance(df, pd.DataFrame):
                df = df.sort_values(by=list(df.columns))
            return df.reset_index(drop=True)
        actual = sort_and_drop_index(actual)
        expected = sort_and_drop_index(expected)
        if isinstance(expected, pd.Series):
            pd.testing.assert_series_equal(actual, expected)
        elif isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected)

    def _evaluate(self, func, *args, distributed=True):
        if False:
            while True:
                i = 10
        deferred_args = [frame_base.DeferredFrame.wrap(expressions.ConstantExpression(arg, arg[0:0])) for arg in args]
        session_type = expressions.PartitioningSession if distributed else expressions.Session
        return session_type({}).evaluate(func(*deferred_args)._expr)

    def test_drop_duplicates_keep_any(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'], 'style': ['cup', 'cup', 'cup', 'pack', 'pack'], 'rating': [4, 4, 3.5, 15, 5]})
        result = self._evaluate(lambda df: df.drop_duplicates(keep='any'), df)
        self.assert_frame_data_equivalent(result, df.drop_duplicates())

    def test_drop_duplicates_keep_any_subset(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'], 'style': ['cup', 'cup', 'cup', 'pack', 'pack'], 'rating': [4, 4, 3.5, 15, 5]})
        result = self._evaluate(lambda df: df.drop_duplicates(keep='any', subset=['brand']), df)
        self.assertTrue(result.brand.unique)
        self.assert_frame_data_equivalent(result.brand, df.drop_duplicates(subset=['brand']).brand)

    def test_series_drop_duplicates_keep_any(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'], 'style': ['cup', 'cup', 'cup', 'pack', 'pack'], 'rating': [4, 4, 3.5, 15, 5]})
        result = self._evaluate(lambda df: df.brand.drop_duplicates(keep='any'), df)
        self.assert_frame_data_equivalent(result, df.brand.drop_duplicates())

    def test_duplicated_keep_any(self):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'], 'style': ['cup', 'cup', 'cup', 'pack', 'pack'], 'rating': [4, 4, 3.5, 15, 5]})
        result = self._evaluate(lambda df: df.duplicated(keep='any'), df)
        self.assert_frame_data_equivalent(result, df.duplicated())

    def test_get_dummies_not_categoricaldtype(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(frame_base.WontImplementError, 'get_dummies\\(\\) of non-categorical type is not supported'):
            s = pd.Series(['a ,b', 'a', 'a, d'])
            self._evaluate(lambda s: s.str.get_dummies(','), s)
        with self.assertRaisesRegex(frame_base.WontImplementError, 'get_dummies\\(\\) of non-categorical type is not supported'):
            s = pd.Series([True, False, False, True])
            self._evaluate(lambda s: s.str.get_dummies(), s)

    def test_get_dummies_comma_separator(self):
        if False:
            while True:
                i = 10
        s = pd.Series(['a ,b', 'a', 'a, d', 'c'])
        s = s.astype(pd.CategoricalDtype(categories=['a ,b', 'c', 'b', 'a,d']))
        result = self._evaluate(lambda s: s.str.get_dummies(','), s)
        self.assert_frame_data_equivalent(result, s.str.get_dummies(','), check_column_subset=True)

    def test_get_dummies_pandas_doc_example1(self):
        if False:
            for i in range(10):
                print('nop')
        s = pd.Series(['a|b', 'a', 'a|c'])
        s = s.astype(pd.CategoricalDtype(categories=['a|b', 'a', 'a|c']))
        result = self._evaluate(lambda s: s.str.get_dummies(), s)
        self.assert_frame_data_equivalent(result, s.str.get_dummies(), check_column_subset=True)

    def test_get_dummies_pandas_doc_example2(self):
        if False:
            i = 10
            return i + 15
        s = pd.Series(['a|b', np.nan, 'a|c'])
        s = s.astype(pd.CategoricalDtype(categories=['a|b', 'a|c']))
        result = self._evaluate(lambda s: s.str.get_dummies(), s)
        self.assert_frame_data_equivalent(result, s.str.get_dummies(), check_column_subset=True)

    def test_get_dummies_pass_nan_as_category(self):
        if False:
            for i in range(10):
                print('nop')
        s = pd.Series(['a|b', 'b|c', 'a|c', 'c', 'd'])
        s = s.astype(pd.CategoricalDtype(categories=['a', 'b', 'c', 'nan']))
        result = self._evaluate(lambda s: s.str.get_dummies(), s)
        self.assert_frame_data_equivalent(result, s.str.get_dummies(), check_column_subset=True)

    def test_get_dummies_bools_casted_to_string(self):
        if False:
            while True:
                i = 10
        s = pd.Series([True, False, False, True]).astype('str')
        s = s.astype(pd.CategoricalDtype(categories=['True', 'False']))
        result = self._evaluate(lambda s: s.str.get_dummies(), s)
        self.assert_frame_data_equivalent(result, s.str.get_dummies(), check_column_subset=True)

    def test_nsmallest_any(self):
        if False:
            return 10
        df = pd.DataFrame({'population': [59000000, 65000000, 434000, 434000, 434000, 337000, 337000, 11300, 11300], 'GDP': [1937894, 2583560, 12011, 4520, 12128, 17036, 182, 38, 311], 'alpha-2': ['IT', 'FR', 'MT', 'MV', 'BN', 'IS', 'NR', 'TV', 'AI']}, index=['Italy', 'France', 'Malta', 'Maldives', 'Brunei', 'Iceland', 'Nauru', 'Tuvalu', 'Anguilla'])
        result = self._evaluate(lambda df: df.population.nsmallest(3, keep='any'), df)
        self.assert_frame_data_equivalent(result, df.population.nsmallest(3))

    def test_nlargest_any(self):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'population': [59000000, 65000000, 434000, 434000, 434000, 337000, 337000, 11300, 11300], 'GDP': [1937894, 2583560, 12011, 4520, 12128, 17036, 182, 38, 311], 'alpha-2': ['IT', 'FR', 'MT', 'MV', 'BN', 'IS', 'NR', 'TV', 'AI']}, index=['Italy', 'France', 'Malta', 'Maldives', 'Brunei', 'Iceland', 'Nauru', 'Tuvalu', 'Anguilla'])
        result = self._evaluate(lambda df: df.population.nlargest(3, keep='any'), df)
        self.assert_frame_data_equivalent(result, df.population.nlargest(3))

    def test_pivot_pandas_example2(self):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'], 'bar': ['A', 'B', 'C', 'A', 'B', 'C'], 'baz': [1, 2, 3, 4, 5, 6], 'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
        df['bar'] = df['bar'].astype(pd.CategoricalDtype(categories=['A', 'B', 'C']))
        result = self._evaluate(lambda df: df.pivot(index='foo', columns='bar'), df)
        self.assert_frame_data_equivalent(result['baz'].apply(pd.to_numeric), df.pivot(index='foo', columns='bar')['baz'])

    def test_sample(self):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'population': [59000000, 65000000, 434000, 434000, 434000, 337000, 337000, 11300, 11300], 'GDP': [1937894, 2583560, 12011, 4520, 12128, 17036, 182, 38, 311], 'alpha-2': ['IT', 'FR', 'MT', 'MV', 'BN', 'IS', 'NR', 'TV', 'AI']}, index=['Italy', 'France', 'Malta', 'Maldives', 'Brunei', 'Iceland', 'Nauru', 'Tuvalu', 'Anguilla'])
        result = self._evaluate(lambda df: df.sample(n=3), df)
        self.assertEqual(len(result), 3)
        series_result = self._evaluate(lambda df: df.GDP.sample(n=3), df)
        self.assertEqual(len(series_result), 3)
        self.assertEqual(series_result.name, 'GDP')

    def test_sample_with_weights(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'population': [59000000, 65000000, 434000, 434000, 434000, 337000, 337000, 11300, 11300], 'GDP': [1937894, 2583560, 12011, 4520, 12128, 17036, 182, 38, 311], 'alpha-2': ['IT', 'FR', 'MT', 'MV', 'BN', 'IS', 'NR', 'TV', 'AI']}, index=['Italy', 'France', 'Malta', 'Maldives', 'Brunei', 'Iceland', 'Nauru', 'Tuvalu', 'Anguilla'])
        weights = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1], index=df.index)
        result = self._evaluate(lambda df, weights: df.sample(n=2, weights=weights), df, weights)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result.index), set(['Tuvalu', 'Anguilla']))
        series_result = self._evaluate(lambda df, weights: df.GDP.sample(n=2, weights=weights), df, weights)
        self.assertEqual(len(series_result), 2)
        self.assertEqual(series_result.name, 'GDP')
        self.assertEqual(set(series_result.index), set(['Tuvalu', 'Anguilla']))

    def test_sample_with_missing_weights(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'population': [59000000, 65000000, 434000, 434000, 434000, 337000, 337000, 11300, 11300], 'GDP': [1937894, 2583560, 12011, 4520, 12128, 17036, 182, 38, 311], 'alpha-2': ['IT', 'FR', 'MT', 'MV', 'BN', 'IS', 'NR', 'TV', 'AI']}, index=['Italy', 'France', 'Malta', 'Maldives', 'Brunei', 'Iceland', 'Nauru', 'Tuvalu', 'Anguilla'])
        weights = pd.Series([0.1, 0.01, np.nan, 0], index=['Nauru', 'Iceland', 'Anguilla', 'Italy'])
        result = self._evaluate(lambda df, weights: df.sample(n=2, weights=weights), df, weights)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result.index), set(['Nauru', 'Iceland']))
        series_result = self._evaluate(lambda df, weights: df.GDP.sample(n=2, weights=weights), df, weights)
        self.assertEqual(len(series_result), 2)
        self.assertEqual(series_result.name, 'GDP')
        self.assertEqual(set(series_result.index), set(['Nauru', 'Iceland']))

    def test_sample_with_weights_distribution(self):
        if False:
            for i in range(10):
                print('nop')
        target_prob = 0.25
        num_samples = 100
        num_targets = 200
        num_other_elements = 10000
        target_weight = target_prob / num_targets
        other_weight = (1 - target_prob) / num_other_elements
        self.assertTrue(target_weight > other_weight * 10, 'weights too close')
        result = self._evaluate(lambda s, weights: s.sample(n=num_samples, weights=weights).sum(), pd.Series([1] * num_targets + [0] * num_other_elements), pd.Series([target_weight] * num_targets + [other_weight] * num_other_elements))
        expected = num_samples * target_prob
        self.assertTrue(expected / 3 < result < expected * 2, (expected, result))

    def test_split_pandas_examples_no_expand(self):
        if False:
            return 10
        s = pd.Series(['this is a regular sentence', 'https://docs.python.org/3/tutorial/index.html', np.nan])
        result = self._evaluate(lambda s: s.str.split(), s)
        self.assert_frame_data_equivalent(result, s.str.split())
        result = self._evaluate(lambda s: s.str.rsplit(), s)
        self.assert_frame_data_equivalent(result, s.str.rsplit())
        result = self._evaluate(lambda s: s.str.split(n=2), s)
        self.assert_frame_data_equivalent(result, s.str.split(n=2))
        result = self._evaluate(lambda s: s.str.rsplit(n=2), s)
        self.assert_frame_data_equivalent(result, s.str.rsplit(n=2))
        result = self._evaluate(lambda s: s.str.split(pat='/'), s)
        self.assert_frame_data_equivalent(result, s.str.split(pat='/'))

    def test_split_pandas_examples_expand_not_categorical(self):
        if False:
            while True:
                i = 10
        s = pd.Series(['this is a regular sentence', 'https://docs.python.org/3/tutorial/index.html', np.nan])
        with self.assertRaisesRegex(frame_base.WontImplementError, 'split\\(\\) of non-categorical type is not supported'):
            self._evaluate(lambda s: s.str.split(expand=True), s)
        with self.assertRaisesRegex(frame_base.WontImplementError, 'rsplit\\(\\) of non-categorical type is not supported'):
            self._evaluate(lambda s: s.str.rsplit(expand=True), s)

    def test_split_pandas_examples_expand_pat_is_string_literal1(self):
        if False:
            while True:
                i = 10
        s = pd.Series(['this is a regular sentence', 'https://docs.python.org/3/tutorial/index.html', np.nan])
        s = s.astype(pd.CategoricalDtype(categories=['this is a regular sentence', 'https://docs.python.org/3/tutorial/index.html']))
        result = self._evaluate(lambda s: s.str.split(expand=True), s)
        self.assert_frame_data_equivalent(result, s.str.split(expand=True))
        result = self._evaluate(lambda s: s.str.rsplit('/', n=1, expand=True), s)
        self.assert_frame_data_equivalent(result, s.str.rsplit('/', n=1, expand=True))

    @unittest.skipIf(PD_VERSION < (1, 4), 'regex arg is new in pandas 1.4')
    def test_split_pandas_examples_expand_pat_is_string_literal2(self):
        if False:
            for i in range(10):
                print('nop')
        s = pd.Series(['foojpgbar.jpg']).astype('category')
        s = s.astype(pd.CategoricalDtype(categories=['foojpgbar.jpg']))
        result = self._evaluate(lambda s: s.str.split('.', expand=True), s)
        self.assert_frame_data_equivalent(result, s.str.split('.', expand=True))
        result = self._evaluate(lambda s: s.str.split('\\.jpg', regex=False, expand=True), s)
        self.assert_frame_data_equivalent(result, s.str.split('\\.jpg', regex=False, expand=True))

    @unittest.skipIf(PD_VERSION < (1, 4), 'regex arg is new in pandas 1.4')
    def test_split_pandas_examples_expand_pat_is_regex(self):
        if False:
            while True:
                i = 10
        s = pd.Series(['foo and bar plus baz'])
        s = s.astype(pd.CategoricalDtype(categories=['foo and bar plus baz']))
        result = self._evaluate(lambda s: s.str.split('and|plus', expand=True), s)
        self.assert_frame_data_equivalent(result, s.str.split('and|plus', expand=True))
        s = pd.Series(['foojpgbar.jpg']).astype('category')
        s = s.astype(pd.CategoricalDtype(categories=['foojpgbar.jpg']))
        result = self._evaluate(lambda s: s.str.split('\\.jpg', expand=True), s)
        self.assert_frame_data_equivalent(result, s.str.split('\\.jpg', expand=True))
        result = self._evaluate(lambda s: s.str.split('\\.jpg', regex=True, expand=True), s)
        self.assert_frame_data_equivalent(result, s.str.split('\\.jpg', regex=True, expand=True))
        result = self._evaluate(lambda s: s.str.split(re.compile('\\.jpg'), expand=True), s)
        self.assert_frame_data_equivalent(result, s.str.split(re.compile('\\.jpg'), expand=True))

    @unittest.skipIf(PD_VERSION < (1, 4), 'regex arg is new in pandas 1.4')
    def test_split_pat_is_regex(self):
        if False:
            return 10
        s = pd.Series(['foojpgbar.jpg']).astype('category')
        s = s.astype(pd.CategoricalDtype(categories=['foojpgbar.jpg']))
        result = self._evaluate(lambda s: s.str.split('\\.jpg', regex=True, expand=False), s)
        self.assert_frame_data_equivalent(result, s.str.split('\\.jpg', regex=True, expand=False))

    def test_astype_categorical_rejected(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'A': np.arange(6), 'B': list('aabbca')})
        with self.assertRaisesRegex(frame_base.WontImplementError, "astype\\(dtype='category'\\)"):
            self._evaluate(lambda df: df.B.astype('category'), df)

class AllowNonParallelTest(unittest.TestCase):

    def _use_non_parallel_operation(self):
        if False:
            for i in range(10):
                print('nop')
        _ = frame_base.DeferredFrame.wrap(expressions.PlaceholderExpression(pd.Series([1, 2, 3]))).replace('a', 'b', limit=1)

    def test_disallow_non_parallel(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(expressions.NonParallelOperation):
            self._use_non_parallel_operation()

    def test_allow_non_parallel_in_context(self):
        if False:
            i = 10
            return i + 15
        with beam.dataframe.allow_non_parallel_operations():
            self._use_non_parallel_operation()

    def test_allow_non_parallel_nesting(self):
        if False:
            i = 10
            return i + 15
        with beam.dataframe.allow_non_parallel_operations():
            self._use_non_parallel_operation()
            with beam.dataframe.allow_non_parallel_operations(False):
                with self.assertRaises(expressions.NonParallelOperation):
                    self._use_non_parallel_operation()
            self._use_non_parallel_operation()
        with self.assertRaises(expressions.NonParallelOperation):
            self._use_non_parallel_operation()

class ConstructionTimeTest(unittest.TestCase):
    """Tests for operations that can be executed eagerly."""
    DF = pd.DataFrame({'str_col': ['foo', 'bar'] * 3, 'int_col': [1, 2] * 3, 'flt_col': [1.1, 2.2] * 3, 'cat_col': pd.Series(list('aabbca'), dtype='category'), 'datetime_col': pd.Series(pd.date_range('1/1/2000', periods=6, freq='m', tz='America/Los_Angeles'))})
    DEFERRED_DF = frame_base.DeferredFrame.wrap(expressions.PlaceholderExpression(DF.iloc[:0]))

    def _run_test(self, fn):
        if False:
            while True:
                i = 10
        expected = fn(self.DF)
        actual = fn(self.DEFERRED_DF)
        if isinstance(expected, pd.Index):
            pd.testing.assert_index_equal(expected, actual)
        elif isinstance(expected, pd.Series):
            pd.testing.assert_series_equal(expected, actual)
        elif isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(expected, actual)
        else:
            self.assertEqual(expected, actual)

    @parameterized.expand(DF.columns)
    def test_series_name(self, col_name):
        if False:
            return 10
        self._run_test(lambda df: df[col_name].name)

    @parameterized.expand(DF.columns)
    def test_series_dtype(self, col_name):
        if False:
            while True:
                i = 10
        self._run_test(lambda df: df[col_name].dtype)
        self._run_test(lambda df: df[col_name].dtypes)

    def test_dataframe_columns(self):
        if False:
            i = 10
            return i + 15
        self._run_test(lambda df: list(df.columns))

    def test_dataframe_dtypes(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_test(lambda df: list(df.dtypes))

    def test_categories(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_test(lambda df: df.cat_col.cat.categories)

    def test_categorical_ordered(self):
        if False:
            print('Hello World!')
        self._run_test(lambda df: df.cat_col.cat.ordered)

    def test_groupby_ndim(self):
        if False:
            print('Hello World!')
        self._run_test(lambda df: df.groupby('int_col').ndim)

    def test_groupby_project_ndim(self):
        if False:
            while True:
                i = 10
        self._run_test(lambda df: df.groupby('int_col').flt_col.ndim)
        self._run_test(lambda df: df.groupby('int_col')[['flt_col', 'str_col']].ndim)

    def test_get_column_default_None(self):
        if False:
            i = 10
            return i + 15
        self._run_test(lambda df: df.get('FOO'))

    def test_datetime_tz(self):
        if False:
            return 10
        self._run_test(lambda df: df.datetime_col.dt.tz)

class DocstringTest(unittest.TestCase):

    @parameterized.expand([(frames.DeferredDataFrame, pd.DataFrame), (frames.DeferredSeries, pd.Series), (frames._DeferredStringMethods, pd.Series.str), (frames._DeferredCategoricalMethods, pd.core.arrays.categorical.CategoricalAccessor), (frames.DeferredGroupBy, pd.core.groupby.generic.DataFrameGroupBy), (frames._DeferredGroupByCols, pd.core.groupby.generic.DataFrameGroupBy), (frames._DeferredDatetimeMethods, pd.core.indexes.accessors.DatetimeProperties)])
    def test_docs_defined(self, beam_type, pd_type):
        if False:
            while True:
                i = 10
        beam_attrs = set(dir(beam_type))
        pd_attrs = set(dir(pd_type))
        docstring_required = sorted([attr for attr in beam_attrs.intersection(pd_attrs) if getattr(pd_type, attr).__doc__ and (not attr.startswith('_'))])
        docstring_missing = [attr for attr in docstring_required if not getattr(beam_type, attr).__doc__]
        self.assertTrue(len(docstring_missing) == 0, f'{beam_type.__name__} is missing a docstring for {len(docstring_missing)}/{len(docstring_required)} ({len(docstring_missing) / len(docstring_required):%}) operations:\n{docstring_missing}')

class ReprTest(unittest.TestCase):

    def test_basic_dataframe(self):
        if False:
            return 10
        df = frame_base.DeferredFrame.wrap(expressions.ConstantExpression(GROUPBY_DF))
        self.assertEqual(repr(df), "DeferredDataFrame(columns=['group', 'foo', 'bar', 'baz', 'bool', 'str'], index=<unnamed>)")

    def test_dataframe_with_named_index(self):
        if False:
            i = 10
            return i + 15
        df = frame_base.DeferredFrame.wrap(expressions.ConstantExpression(GROUPBY_DF.set_index('group')))
        self.assertEqual(repr(df), "DeferredDataFrame(columns=['foo', 'bar', 'baz', 'bool', 'str'], index='group')")

    def test_dataframe_with_partial_named_index(self):
        if False:
            print('Hello World!')
        df = frame_base.DeferredFrame.wrap(expressions.ConstantExpression(GROUPBY_DF.set_index([GROUPBY_DF.index, 'group'])))
        self.assertEqual(repr(df), "DeferredDataFrame(columns=['foo', 'bar', 'baz', 'bool', 'str'], indexes=[<unnamed>, 'group'])")

    def test_dataframe_with_named_multi_index(self):
        if False:
            for i in range(10):
                print('nop')
        df = frame_base.DeferredFrame.wrap(expressions.ConstantExpression(GROUPBY_DF.set_index(['str', 'group'])))
        self.assertEqual(repr(df), "DeferredDataFrame(columns=['foo', 'bar', 'baz', 'bool'], indexes=['str', 'group'])")

    def test_dataframe_with_multiple_column_levels(self):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'foofoofoo': ['one', 'one', 'one', 'two', 'two', 'two'], 'barbar': ['A', 'B', 'C', 'A', 'B', 'C'], 'bazzy': [1, 2, 3, 4, 5, 6], 'zoop': ['x', 'y', 'z', 'q', 'w', 't']})
        df = df.pivot(index='foofoofoo', columns='barbar')
        df = frame_base.DeferredFrame.wrap(expressions.ConstantExpression(df))
        self.assertEqual(repr(df), "DeferredDataFrame(columns=[('bazzy', 'A'), ('bazzy', 'B'), ('bazzy', 'C'), ('zoop', 'A'), ('zoop', 'B'), ('zoop', 'C')], index='foofoofoo')")

    def test_dataframe_with_multiple_column_and_multiple_index_levels(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'foofoofoo': ['one', 'one', 'one', 'two', 'two', 'two'], 'barbar': ['A', 'B', 'C', 'A', 'B', 'C'], 'bazzy': [1, 2, 3, 4, 5, 6], 'zoop': ['x', 'y', 'z', 'q', 'w', 't']})
        df = df.pivot(index='foofoofoo', columns='barbar')
        df.index = [['a', 'b'], df.index]
        df = frame_base.DeferredFrame.wrap(expressions.ConstantExpression(df))
        self.assertEqual(repr(df), "DeferredDataFrame(columns=[('bazzy', 'A'), ('bazzy', 'B'), ('bazzy', 'C'), ('zoop', 'A'), ('zoop', 'B'), ('zoop', 'C')], indexes=[<unnamed>, 'foofoofoo'])")

    def test_basic_series(self):
        if False:
            i = 10
            return i + 15
        df = frame_base.DeferredFrame.wrap(expressions.ConstantExpression(GROUPBY_DF['bool']))
        self.assertEqual(repr(df), "DeferredSeries(name='bool', dtype=bool, index=<unnamed>)")

    def test_series_with_named_index(self):
        if False:
            print('Hello World!')
        df = frame_base.DeferredFrame.wrap(expressions.ConstantExpression(GROUPBY_DF.set_index('group')['str']))
        self.assertEqual(repr(df), "DeferredSeries(name='str', dtype=object, index='group')")

    def test_series_with_partial_named_index(self):
        if False:
            while True:
                i = 10
        df = frame_base.DeferredFrame.wrap(expressions.ConstantExpression(GROUPBY_DF.set_index([GROUPBY_DF.index, 'group'])['bar']))
        self.assertEqual(repr(df), "DeferredSeries(name='bar', dtype=float64, indexes=[<unnamed>, 'group'])")

    def test_series_with_named_multi_index(self):
        if False:
            return 10
        df = frame_base.DeferredFrame.wrap(expressions.ConstantExpression(GROUPBY_DF.set_index(['str', 'group'])['baz']))
        self.assertEqual(repr(df), "DeferredSeries(name='baz', dtype=float64, indexes=['str', 'group'])")

@unittest.skipIf(not ie.current_env().is_interactive_ready, '[interactive] dependency is not installed.')
@isolated_env
class InteractiveDataFrameTest(unittest.TestCase):

    def test_collect_merged_dataframes(self):
        if False:
            while True:
                i = 10
        p = beam.Pipeline(InteractiveRunner())
        pcoll_1 = p | 'Create data 1' >> beam.Create([(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]) | 'To rows 1' >> beam.Select(col_1=lambda x: x[0], col_2=lambda x: x[1])
        df_1 = to_dataframe(pcoll_1)
        pcoll_2 = p | 'Create data 2' >> beam.Create([(5, 'e'), (6, 'f'), (7, 'g'), (8, 'h')]) | 'To rows 2' >> beam.Select(col_3=lambda x: x[0], col_4=lambda x: x[1])
        df_2 = to_dataframe(pcoll_2)
        df_merged = df_1.merge(df_2, left_index=True, right_index=True)
        pd_df = ib.collect(df_merged).sort_values(by='col_1')
        self.assertEqual(pd_df.shape, (4, 4))
        self.assertEqual(list(pd_df['col_1']), [1, 2, 3, 4])
        self.assertEqual(list(pd_df['col_2']), ['a', 'b', 'c', 'd'])
        self.assertEqual(list(pd_df['col_3']), [5, 6, 7, 8])
        self.assertEqual(list(pd_df['col_4']), ['e', 'f', 'g', 'h'])
if __name__ == '__main__':
    unittest.main()