import string
import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series, array, concat, date_range, merge, merge_asof
from .pandas_vb_common import tm
try:
    from pandas import merge_ordered
except ImportError:
    from pandas import ordered_merge as merge_ordered

class Concat:
    params = [0, 1]
    param_names = ['axis']

    def setup(self, axis):
        if False:
            for i in range(10):
                print('nop')
        N = 1000
        s = Series(N, index=tm.makeStringIndex(N))
        self.series = [s[i:-i] for i in range(1, 10)] * 50
        self.small_frames = [DataFrame(np.random.randn(5, 4))] * 1000
        df = DataFrame({'A': range(N)}, index=date_range('20130101', periods=N, freq='s'))
        self.empty_left = [DataFrame(), df]
        self.empty_right = [df, DataFrame()]
        self.mixed_ndims = [df, df.head(N // 2)]

    def time_concat_series(self, axis):
        if False:
            return 10
        concat(self.series, axis=axis, sort=False)

    def time_concat_small_frames(self, axis):
        if False:
            return 10
        concat(self.small_frames, axis=axis)

    def time_concat_empty_right(self, axis):
        if False:
            while True:
                i = 10
        concat(self.empty_right, axis=axis)

    def time_concat_empty_left(self, axis):
        if False:
            return 10
        concat(self.empty_left, axis=axis)

    def time_concat_mixed_ndims(self, axis):
        if False:
            print('Hello World!')
        concat(self.mixed_ndims, axis=axis)

class ConcatDataFrames:
    params = ([0, 1], [True, False])
    param_names = ['axis', 'ignore_index']

    def setup(self, axis, ignore_index):
        if False:
            i = 10
            return i + 15
        frame_c = DataFrame(np.zeros((10000, 200), dtype=np.float32, order='C'))
        self.frame_c = [frame_c] * 20
        frame_f = DataFrame(np.zeros((10000, 200), dtype=np.float32, order='F'))
        self.frame_f = [frame_f] * 20

    def time_c_ordered(self, axis, ignore_index):
        if False:
            i = 10
            return i + 15
        concat(self.frame_c, axis=axis, ignore_index=ignore_index)

    def time_f_ordered(self, axis, ignore_index):
        if False:
            for i in range(10):
                print('nop')
        concat(self.frame_f, axis=axis, ignore_index=ignore_index)

class ConcatIndexDtype:
    params = (['datetime64[ns]', 'int64', 'Int64', 'int64[pyarrow]', 'string[python]', 'string[pyarrow]'], ['monotonic', 'non_monotonic', 'has_na'], [0, 1], [True, False])
    param_names = ['dtype', 'structure', 'axis', 'sort']

    def setup(self, dtype, structure, axis, sort):
        if False:
            while True:
                i = 10
        N = 10000
        if dtype == 'datetime64[ns]':
            vals = date_range('1970-01-01', periods=N)
        elif dtype in ('int64', 'Int64', 'int64[pyarrow]'):
            vals = np.arange(N, dtype=np.int64)
        elif dtype in ('string[python]', 'string[pyarrow]'):
            vals = tm.makeStringIndex(N)
        else:
            raise NotImplementedError
        idx = Index(vals, dtype=dtype)
        if structure == 'monotonic':
            idx = idx.sort_values()
        elif structure == 'non_monotonic':
            idx = idx[::-1]
        elif structure == 'has_na':
            if not idx._can_hold_na:
                raise NotImplementedError
            idx = Index([None], dtype=dtype).append(idx)
        else:
            raise NotImplementedError
        self.series = [Series(i, idx[:-i]) for i in range(1, 6)]

    def time_concat_series(self, dtype, structure, axis, sort):
        if False:
            print('Hello World!')
        concat(self.series, axis=axis, sort=sort)

class Join:
    params = [True, False]
    param_names = ['sort']

    def setup(self, sort):
        if False:
            return 10
        level1 = tm.makeStringIndex(10).values
        level2 = tm.makeStringIndex(1000).values
        codes1 = np.arange(10).repeat(1000)
        codes2 = np.tile(np.arange(1000), 10)
        index2 = MultiIndex(levels=[level1, level2], codes=[codes1, codes2])
        self.df_multi = DataFrame(np.random.randn(len(index2), 4), index=index2, columns=['A', 'B', 'C', 'D'])
        self.key1 = np.tile(level1.take(codes1), 10)
        self.key2 = np.tile(level2.take(codes2), 10)
        self.df = DataFrame({'data1': np.random.randn(100000), 'data2': np.random.randn(100000), 'key1': self.key1, 'key2': self.key2})
        self.df_key1 = DataFrame(np.random.randn(len(level1), 4), index=level1, columns=['A', 'B', 'C', 'D'])
        self.df_key2 = DataFrame(np.random.randn(len(level2), 4), index=level2, columns=['A', 'B', 'C', 'D'])
        shuf = np.arange(100000)
        np.random.shuffle(shuf)
        self.df_shuf = self.df.reindex(self.df.index[shuf])

    def time_join_dataframe_index_multi(self, sort):
        if False:
            return 10
        self.df.join(self.df_multi, on=['key1', 'key2'], sort=sort)

    def time_join_dataframe_index_single_key_bigger(self, sort):
        if False:
            print('Hello World!')
        self.df.join(self.df_key2, on='key2', sort=sort)

    def time_join_dataframe_index_single_key_small(self, sort):
        if False:
            print('Hello World!')
        self.df.join(self.df_key1, on='key1', sort=sort)

    def time_join_dataframe_index_shuffle_key_bigger_sort(self, sort):
        if False:
            for i in range(10):
                print('nop')
        self.df_shuf.join(self.df_key2, on='key2', sort=sort)

    def time_join_dataframes_cross(self, sort):
        if False:
            return 10
        self.df.loc[:2000].join(self.df_key1, how='cross', sort=sort)

class JoinIndex:

    def setup(self):
        if False:
            print('Hello World!')
        N = 5000
        self.left = DataFrame(np.random.randint(1, N / 50, (N, 2)), columns=['jim', 'joe'])
        self.right = DataFrame(np.random.randint(1, N / 50, (N, 2)), columns=['jolie', 'jolia']).set_index('jolie')

    def time_left_outer_join_index(self):
        if False:
            print('Hello World!')
        self.left.join(self.right, on='jim')

class JoinMultiindexSubset:

    def setup(self):
        if False:
            while True:
                i = 10
        N = 100000
        mi1 = MultiIndex.from_arrays([np.arange(N)] * 4, names=['a', 'b', 'c', 'd'])
        mi2 = MultiIndex.from_arrays([np.arange(N)] * 2, names=['a', 'b'])
        self.left = DataFrame({'col1': 1}, index=mi1)
        self.right = DataFrame({'col2': 2}, index=mi2)

    def time_join_multiindex_subset(self):
        if False:
            i = 10
            return i + 15
        self.left.join(self.right)

class JoinEmpty:

    def setup(self):
        if False:
            return 10
        N = 100000
        self.df = DataFrame({'A': np.arange(N)})
        self.df_empty = DataFrame(columns=['B', 'C'], dtype='int64')

    def time_inner_join_left_empty(self):
        if False:
            i = 10
            return i + 15
        self.df_empty.join(self.df, how='inner')

    def time_inner_join_right_empty(self):
        if False:
            for i in range(10):
                print('nop')
        self.df.join(self.df_empty, how='inner')

class JoinNonUnique:

    def setup(self):
        if False:
            i = 10
            return i + 15
        date_index = date_range('01-Jan-2013', '23-Jan-2013', freq='min')
        daily_dates = date_index.to_period('D').to_timestamp('s', 's')
        self.fracofday = date_index.values - daily_dates.values
        self.fracofday = self.fracofday.astype('timedelta64[ns]')
        self.fracofday = self.fracofday.astype(np.float64) / 86400000000000
        self.fracofday = Series(self.fracofday, daily_dates)
        index = date_range(date_index.min(), date_index.max(), freq='D')
        self.temp = Series(1.0, index)[self.fracofday.index]

    def time_join_non_unique_equal(self):
        if False:
            i = 10
            return i + 15
        self.fracofday * self.temp

class Merge:
    params = [True, False]
    param_names = ['sort']

    def setup(self, sort):
        if False:
            return 10
        N = 10000
        indices = tm.makeStringIndex(N).values
        indices2 = tm.makeStringIndex(N).values
        key = np.tile(indices[:8000], 10)
        key2 = np.tile(indices2[:8000], 10)
        self.left = DataFrame({'key': key, 'key2': key2, 'value': np.random.randn(80000)})
        self.right = DataFrame({'key': indices[2000:], 'key2': indices2[2000:], 'value2': np.random.randn(8000)})
        self.df = DataFrame({'key1': np.tile(np.arange(500).repeat(10), 2), 'key2': np.tile(np.arange(250).repeat(10), 4), 'value': np.random.randn(10000)})
        self.df2 = DataFrame({'key1': np.arange(500), 'value2': np.random.randn(500)})
        self.df3 = self.df[:5000]

    def time_merge_2intkey(self, sort):
        if False:
            return 10
        merge(self.left, self.right, sort=sort)

    def time_merge_dataframe_integer_2key(self, sort):
        if False:
            print('Hello World!')
        merge(self.df, self.df3, sort=sort)

    def time_merge_dataframe_integer_key(self, sort):
        if False:
            while True:
                i = 10
        merge(self.df, self.df2, on='key1', sort=sort)

    def time_merge_dataframe_empty_right(self, sort):
        if False:
            while True:
                i = 10
        merge(self.left, self.right.iloc[:0], sort=sort)

    def time_merge_dataframe_empty_left(self, sort):
        if False:
            for i in range(10):
                print('nop')
        merge(self.left.iloc[:0], self.right, sort=sort)

    def time_merge_dataframes_cross(self, sort):
        if False:
            return 10
        merge(self.left.loc[:2000], self.right.loc[:2000], how='cross', sort=sort)

class MergeEA:
    params = ['Int64', 'Int32', 'Int16', 'UInt64', 'UInt32', 'UInt16', 'Float64', 'Float32']
    param_names = ['dtype']

    def setup(self, dtype):
        if False:
            i = 10
            return i + 15
        N = 10000
        indices = np.arange(1, N)
        key = np.tile(indices[:8000], 10)
        self.left = DataFrame({'key': Series(key, dtype=dtype), 'value': np.random.randn(80000)})
        self.right = DataFrame({'key': Series(indices[2000:], dtype=dtype), 'value2': np.random.randn(7999)})

    def time_merge(self, dtype):
        if False:
            return 10
        merge(self.left, self.right)

class I8Merge:
    params = ['inner', 'outer', 'left', 'right']
    param_names = ['how']

    def setup(self, how):
        if False:
            for i in range(10):
                print('nop')
        (low, high, n) = (-1000, 1000, 10 ** 6)
        self.left = DataFrame(np.random.randint(low, high, (n, 7)), columns=list('ABCDEFG'))
        self.left['left'] = self.left.sum(axis=1)
        self.right = self.left.sample(frac=1).rename({'left': 'right'}, axis=1)
        self.right = self.right.reset_index(drop=True)
        self.right['right'] *= -1

    def time_i8merge(self, how):
        if False:
            while True:
                i = 10
        merge(self.left, self.right, how=how)

class MergeDatetime:
    params = [[('ns', 'ns'), ('ms', 'ms'), ('ns', 'ms')], [None, 'Europe/Brussels']]
    param_names = ['units', 'tz']

    def setup(self, units, tz):
        if False:
            for i in range(10):
                print('nop')
        (unit_left, unit_right) = units
        N = 10000
        keys = Series(date_range('2012-01-01', freq='min', periods=N, tz=tz))
        self.left = DataFrame({'key': keys.sample(N * 10, replace=True).dt.as_unit(unit_left), 'value1': np.random.randn(N * 10)})
        self.right = DataFrame({'key': keys[:8000].dt.as_unit(unit_right), 'value2': np.random.randn(8000)})

    def time_merge(self, units, tz):
        if False:
            while True:
                i = 10
        merge(self.left, self.right)

class MergeCategoricals:

    def setup(self):
        if False:
            while True:
                i = 10
        self.left_object = DataFrame({'X': np.random.choice(range(10), size=(10000,)), 'Y': np.random.choice(['one', 'two', 'three'], size=(10000,))})
        self.right_object = DataFrame({'X': np.random.choice(range(10), size=(10000,)), 'Z': np.random.choice(['jjj', 'kkk', 'sss'], size=(10000,))})
        self.left_cat = self.left_object.assign(Y=self.left_object['Y'].astype('category'))
        self.right_cat = self.right_object.assign(Z=self.right_object['Z'].astype('category'))
        self.left_cat_col = self.left_object.astype({'X': 'category'})
        self.right_cat_col = self.right_object.astype({'X': 'category'})
        self.left_cat_idx = self.left_cat_col.set_index('X')
        self.right_cat_idx = self.right_cat_col.set_index('X')

    def time_merge_object(self):
        if False:
            return 10
        merge(self.left_object, self.right_object, on='X')

    def time_merge_cat(self):
        if False:
            while True:
                i = 10
        merge(self.left_cat, self.right_cat, on='X')

    def time_merge_on_cat_col(self):
        if False:
            while True:
                i = 10
        merge(self.left_cat_col, self.right_cat_col, on='X')

    def time_merge_on_cat_idx(self):
        if False:
            while True:
                i = 10
        merge(self.left_cat_idx, self.right_cat_idx, on='X')

class MergeOrdered:

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        groups = tm.makeStringIndex(10).values
        self.left = DataFrame({'group': groups.repeat(5000), 'key': np.tile(np.arange(0, 10000, 2), 10), 'lvalue': np.random.randn(50000)})
        self.right = DataFrame({'key': np.arange(10000), 'rvalue': np.random.randn(10000)})

    def time_merge_ordered(self):
        if False:
            print('Hello World!')
        merge_ordered(self.left, self.right, on='key', left_by='group')

class MergeAsof:
    params = [['backward', 'forward', 'nearest'], [None, 5]]
    param_names = ['direction', 'tolerance']

    def setup(self, direction, tolerance):
        if False:
            while True:
                i = 10
        one_count = 200000
        two_count = 1000000
        df1 = DataFrame({'time': np.random.randint(0, one_count / 20, one_count), 'key': np.random.choice(list(string.ascii_uppercase), one_count), 'key2': np.random.randint(0, 25, one_count), 'value1': np.random.randn(one_count)})
        df2 = DataFrame({'time': np.random.randint(0, two_count / 20, two_count), 'key': np.random.choice(list(string.ascii_uppercase), two_count), 'key2': np.random.randint(0, 25, two_count), 'value2': np.random.randn(two_count)})
        df1 = df1.sort_values('time')
        df2 = df2.sort_values('time')
        df1['time32'] = np.int32(df1.time)
        df2['time32'] = np.int32(df2.time)
        df1['timeu64'] = np.uint64(df1.time)
        df2['timeu64'] = np.uint64(df2.time)
        self.df1a = df1[['time', 'value1']]
        self.df2a = df2[['time', 'value2']]
        self.df1b = df1[['time', 'key', 'value1']]
        self.df2b = df2[['time', 'key', 'value2']]
        self.df1c = df1[['time', 'key2', 'value1']]
        self.df2c = df2[['time', 'key2', 'value2']]
        self.df1d = df1[['time32', 'value1']]
        self.df2d = df2[['time32', 'value2']]
        self.df1e = df1[['time', 'key', 'key2', 'value1']]
        self.df2e = df2[['time', 'key', 'key2', 'value2']]
        self.df1f = df1[['timeu64', 'value1']]
        self.df2f = df2[['timeu64', 'value2']]

    def time_on_int(self, direction, tolerance):
        if False:
            i = 10
            return i + 15
        merge_asof(self.df1a, self.df2a, on='time', direction=direction, tolerance=tolerance)

    def time_on_int32(self, direction, tolerance):
        if False:
            print('Hello World!')
        merge_asof(self.df1d, self.df2d, on='time32', direction=direction, tolerance=tolerance)

    def time_on_uint64(self, direction, tolerance):
        if False:
            return 10
        merge_asof(self.df1f, self.df2f, on='timeu64', direction=direction, tolerance=tolerance)

    def time_by_object(self, direction, tolerance):
        if False:
            print('Hello World!')
        merge_asof(self.df1b, self.df2b, on='time', by='key', direction=direction, tolerance=tolerance)

    def time_by_int(self, direction, tolerance):
        if False:
            while True:
                i = 10
        merge_asof(self.df1c, self.df2c, on='time', by='key2', direction=direction, tolerance=tolerance)

    def time_multiby(self, direction, tolerance):
        if False:
            while True:
                i = 10
        merge_asof(self.df1e, self.df2e, on='time', by=['key', 'key2'], direction=direction, tolerance=tolerance)

class MergeMultiIndex:
    params = [[('int64', 'int64'), ('datetime64[ns]', 'int64'), ('Int64', 'Int64')], ['left', 'right', 'inner', 'outer']]
    param_names = ['dtypes', 'how']

    def setup(self, dtypes, how):
        if False:
            print('Hello World!')
        n = 100000
        offset = 50000
        mi1 = MultiIndex.from_arrays([array(np.arange(n), dtype=dtypes[0]), array(np.arange(n), dtype=dtypes[1])])
        mi2 = MultiIndex.from_arrays([array(np.arange(offset, n + offset), dtype=dtypes[0]), array(np.arange(offset, n + offset), dtype=dtypes[1])])
        self.df1 = DataFrame({'col1': 1}, index=mi1)
        self.df2 = DataFrame({'col2': 2}, index=mi2)

    def time_merge_sorted_multiindex(self, dtypes, how):
        if False:
            return 10
        df1 = self.df1.copy()
        df2 = self.df2.copy()
        merge(df1, df2, how=how, left_index=True, right_index=True)

class Align:

    def setup(self):
        if False:
            i = 10
            return i + 15
        size = 5 * 10 ** 5
        rng = np.arange(0, 10 ** 13, 10 ** 7)
        stamps = np.datetime64('now').view('i8') + rng
        idx1 = np.sort(np.random.choice(stamps, size, replace=False))
        idx2 = np.sort(np.random.choice(stamps, size, replace=False))
        self.ts1 = Series(np.random.randn(size), idx1)
        self.ts2 = Series(np.random.randn(size), idx2)

    def time_series_align_int64_index(self):
        if False:
            i = 10
            return i + 15
        self.ts1 + self.ts2

    def time_series_align_left_monotonic(self):
        if False:
            return 10
        self.ts1.align(self.ts2, join='left')
from .pandas_vb_common import setup