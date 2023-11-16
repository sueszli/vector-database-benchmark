import string
import numpy as np
from pandas import NA, DataFrame, MultiIndex, RangeIndex, Series, array, date_range
from .pandas_vb_common import tm

class GetLoc:

    def setup(self):
        if False:
            while True:
                i = 10
        self.mi_large = MultiIndex.from_product([np.arange(1000), np.arange(20), list(string.ascii_letters)], names=['one', 'two', 'three'])
        self.mi_med = MultiIndex.from_product([np.arange(1000), np.arange(10), list('A')], names=['one', 'two', 'three'])
        self.mi_small = MultiIndex.from_product([np.arange(100), list('A'), list('A')], names=['one', 'two', 'three'])

    def time_large_get_loc(self):
        if False:
            print('Hello World!')
        self.mi_large.get_loc((999, 19, 'Z'))

    def time_large_get_loc_warm(self):
        if False:
            for i in range(10):
                print('nop')
        for _ in range(1000):
            self.mi_large.get_loc((999, 19, 'Z'))

    def time_med_get_loc(self):
        if False:
            return 10
        self.mi_med.get_loc((999, 9, 'A'))

    def time_med_get_loc_warm(self):
        if False:
            i = 10
            return i + 15
        for _ in range(1000):
            self.mi_med.get_loc((999, 9, 'A'))

    def time_string_get_loc(self):
        if False:
            for i in range(10):
                print('nop')
        self.mi_small.get_loc((99, 'A', 'A'))

    def time_small_get_loc_warm(self):
        if False:
            for i in range(10):
                print('nop')
        for _ in range(1000):
            self.mi_small.get_loc((99, 'A', 'A'))

class GetLocs:

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.mi_large = MultiIndex.from_product([np.arange(1000), np.arange(20), list(string.ascii_letters)], names=['one', 'two', 'three'])
        self.mi_med = MultiIndex.from_product([np.arange(1000), np.arange(10), list('A')], names=['one', 'two', 'three'])
        self.mi_small = MultiIndex.from_product([np.arange(100), list('A'), list('A')], names=['one', 'two', 'three'])

    def time_large_get_locs(self):
        if False:
            for i in range(10):
                print('nop')
        self.mi_large.get_locs([999, 19, 'Z'])

    def time_med_get_locs(self):
        if False:
            while True:
                i = 10
        self.mi_med.get_locs([999, 9, 'A'])

    def time_small_get_locs(self):
        if False:
            print('Hello World!')
        self.mi_small.get_locs([99, 'A', 'A'])

class Duplicates:

    def setup(self):
        if False:
            return 10
        size = 65536
        arrays = [np.random.randint(0, 8192, size), np.random.randint(0, 1024, size)]
        mask = np.random.rand(size) < 0.1
        self.mi_unused_levels = MultiIndex.from_arrays(arrays)
        self.mi_unused_levels = self.mi_unused_levels[mask]

    def time_remove_unused_levels(self):
        if False:
            i = 10
            return i + 15
        self.mi_unused_levels.remove_unused_levels()

class Integer:

    def setup(self):
        if False:
            while True:
                i = 10
        self.mi_int = MultiIndex.from_product([np.arange(1000), np.arange(1000)], names=['one', 'two'])
        self.obj_index = np.array([(0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19)], dtype=object)
        self.other_mi_many_mismatches = MultiIndex.from_tuples([(-7, 41), (-2, 3), (-0.7, 5), (0, 0), (0, 1.5), (0, 340), (0, 1001), (1, -4), (1, 20), (1, 1040), (432, -5), (432, 17), (439, 165.5), (998, -4), (998, 24065), (999, 865.2), (999, 1000), (1045, -843)])

    def time_get_indexer(self):
        if False:
            print('Hello World!')
        self.mi_int.get_indexer(self.obj_index)

    def time_get_indexer_and_backfill(self):
        if False:
            for i in range(10):
                print('nop')
        self.mi_int.get_indexer(self.other_mi_many_mismatches, method='backfill')

    def time_get_indexer_and_pad(self):
        if False:
            return 10
        self.mi_int.get_indexer(self.other_mi_many_mismatches, method='pad')

    def time_is_monotonic(self):
        if False:
            while True:
                i = 10
        self.mi_int.is_monotonic_increasing

class Duplicated:

    def setup(self):
        if False:
            while True:
                i = 10
        (n, k) = (200, 5000)
        levels = [np.arange(n), tm.makeStringIndex(n).values, 1000 + np.arange(n)]
        codes = [np.random.choice(n, k * n) for lev in levels]
        self.mi = MultiIndex(levels=levels, codes=codes)

    def time_duplicated(self):
        if False:
            for i in range(10):
                print('nop')
        self.mi.duplicated()

class Sortlevel:

    def setup(self):
        if False:
            while True:
                i = 10
        n = 1182720
        (low, high) = (-4096, 4096)
        arrs = [np.repeat(np.random.randint(low, high, n // k), k) for k in [11, 7, 5, 3, 1]]
        self.mi_int = MultiIndex.from_arrays(arrs)[np.random.permutation(n)]
        a = np.repeat(np.arange(100), 1000)
        b = np.tile(np.arange(1000), 100)
        self.mi = MultiIndex.from_arrays([a, b])
        self.mi = self.mi.take(np.random.permutation(np.arange(100000)))

    def time_sortlevel_int64(self):
        if False:
            return 10
        self.mi_int.sortlevel()

    def time_sortlevel_zero(self):
        if False:
            i = 10
            return i + 15
        self.mi.sortlevel(0)

    def time_sortlevel_one(self):
        if False:
            while True:
                i = 10
        self.mi.sortlevel(1)

class SortValues:
    params = ['int64', 'Int64']
    param_names = ['dtype']

    def setup(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        a = array(np.tile(np.arange(100), 1000), dtype=dtype)
        b = array(np.tile(np.arange(1000), 100), dtype=dtype)
        self.mi = MultiIndex.from_arrays([a, b])

    def time_sort_values(self, dtype):
        if False:
            i = 10
            return i + 15
        self.mi.sort_values()

class Values:

    def setup_cache(self):
        if False:
            return 10
        level1 = range(1000)
        level2 = date_range(start='1/1/2012', periods=100)
        mi = MultiIndex.from_product([level1, level2])
        return mi

    def time_datetime_level_values_copy(self, mi):
        if False:
            for i in range(10):
                print('nop')
        mi.copy().values

    def time_datetime_level_values_sliced(self, mi):
        if False:
            print('Hello World!')
        mi[:10].values

class CategoricalLevel:

    def setup(self):
        if False:
            while True:
                i = 10
        self.df = DataFrame({'a': np.arange(1000000, dtype=np.int32), 'b': np.arange(1000000, dtype=np.int64), 'c': np.arange(1000000, dtype=float)}).astype({'a': 'category', 'b': 'category'})

    def time_categorical_level(self):
        if False:
            print('Hello World!')
        self.df.set_index(['a', 'b'])

class Equals:

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        idx_large_fast = RangeIndex(100000)
        idx_small_slow = date_range(start='1/1/2012', periods=1)
        self.mi_large_slow = MultiIndex.from_product([idx_large_fast, idx_small_slow])
        self.idx_non_object = RangeIndex(1)

    def time_equals_non_object_index(self):
        if False:
            print('Hello World!')
        self.mi_large_slow.equals(self.idx_non_object)

class SetOperations:
    params = [('monotonic', 'non_monotonic'), ('datetime', 'int', 'string', 'ea_int'), ('intersection', 'union', 'symmetric_difference'), (False, None)]
    param_names = ['index_structure', 'dtype', 'method', 'sort']

    def setup(self, index_structure, dtype, method, sort):
        if False:
            while True:
                i = 10
        N = 10 ** 5
        level1 = range(1000)
        level2 = date_range(start='1/1/2000', periods=N // 1000)
        dates_left = MultiIndex.from_product([level1, level2])
        level2 = range(N // 1000)
        int_left = MultiIndex.from_product([level1, level2])
        level2 = tm.makeStringIndex(N // 1000).values
        str_left = MultiIndex.from_product([level1, level2])
        level2 = range(N // 1000)
        ea_int_left = MultiIndex.from_product([level1, Series(level2, dtype='Int64')])
        data = {'datetime': dates_left, 'int': int_left, 'string': str_left, 'ea_int': ea_int_left}
        if index_structure == 'non_monotonic':
            data = {k: mi[::-1] for (k, mi) in data.items()}
        data = {k: {'left': mi, 'right': mi[:-1]} for (k, mi) in data.items()}
        self.left = data[dtype]['left']
        self.right = data[dtype]['right']

    def time_operation(self, index_structure, dtype, method, sort):
        if False:
            i = 10
            return i + 15
        getattr(self.left, method)(self.right, sort=sort)

class Difference:
    params = [('datetime', 'int', 'string', 'ea_int')]
    param_names = ['dtype']

    def setup(self, dtype):
        if False:
            while True:
                i = 10
        N = 10 ** 4 * 2
        level1 = range(1000)
        level2 = date_range(start='1/1/2000', periods=N // 1000)
        dates_left = MultiIndex.from_product([level1, level2])
        level2 = range(N // 1000)
        int_left = MultiIndex.from_product([level1, level2])
        level2 = Series(range(N // 1000), dtype='Int64')
        level2[0] = NA
        ea_int_left = MultiIndex.from_product([level1, level2])
        level2 = tm.makeStringIndex(N // 1000).values
        str_left = MultiIndex.from_product([level1, level2])
        data = {'datetime': dates_left, 'int': int_left, 'ea_int': ea_int_left, 'string': str_left}
        data = {k: {'left': mi, 'right': mi[:5]} for (k, mi) in data.items()}
        self.left = data[dtype]['left']
        self.right = data[dtype]['right']

    def time_difference(self, dtype):
        if False:
            while True:
                i = 10
        self.left.difference(self.right)

class Unique:
    params = [(('Int64', NA), ('int64', 0))]
    param_names = ['dtype_val']

    def setup(self, dtype_val):
        if False:
            print('Hello World!')
        level = Series([1, 2, dtype_val[1], dtype_val[1]] + list(range(1000000)), dtype=dtype_val[0])
        self.midx = MultiIndex.from_arrays([level, level])
        level_dups = Series([1, 2, dtype_val[1], dtype_val[1]] + list(range(500000)) * 2, dtype=dtype_val[0])
        self.midx_dups = MultiIndex.from_arrays([level_dups, level_dups])

    def time_unique(self, dtype_val):
        if False:
            i = 10
            return i + 15
        self.midx.unique()

    def time_unique_dups(self, dtype_val):
        if False:
            print('Hello World!')
        self.midx_dups.unique()

class Isin:
    params = [('string', 'int', 'datetime')]
    param_names = ['dtype']

    def setup(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        N = 10 ** 5
        level1 = range(1000)
        level2 = date_range(start='1/1/2000', periods=N // 1000)
        dates_midx = MultiIndex.from_product([level1, level2])
        level2 = range(N // 1000)
        int_midx = MultiIndex.from_product([level1, level2])
        level2 = tm.makeStringIndex(N // 1000).values
        str_midx = MultiIndex.from_product([level1, level2])
        data = {'datetime': dates_midx, 'int': int_midx, 'string': str_midx}
        self.midx = data[dtype]
        self.values_small = self.midx[:100]
        self.values_large = self.midx[100:]

    def time_isin_small(self, dtype):
        if False:
            return 10
        self.midx.isin(self.values_small)

    def time_isin_large(self, dtype):
        if False:
            while True:
                i = 10
        self.midx.isin(self.values_large)

class Putmask:

    def setup(self):
        if False:
            print('Hello World!')
        N = 10 ** 5
        level1 = range(1000)
        level2 = date_range(start='1/1/2000', periods=N // 1000)
        self.midx = MultiIndex.from_product([level1, level2])
        level1 = range(1000, 2000)
        self.midx_values = MultiIndex.from_product([level1, level2])
        level2 = date_range(start='1/1/2010', periods=N // 1000)
        self.midx_values_different = MultiIndex.from_product([level1, level2])
        self.mask = np.array([True, False] * (N // 2))

    def time_putmask(self):
        if False:
            for i in range(10):
                print('nop')
        self.midx.putmask(self.mask, self.midx_values)

    def time_putmask_all_different(self):
        if False:
            for i in range(10):
                print('nop')
        self.midx.putmask(self.mask, self.midx_values_different)

class Append:
    params = ['datetime64[ns]', 'int64', 'string']
    param_names = ['dtype']

    def setup(self, dtype):
        if False:
            i = 10
            return i + 15
        N1 = 1000
        N2 = 500
        left_level1 = range(N1)
        right_level1 = range(N1, N1 + N1)
        if dtype == 'datetime64[ns]':
            level2 = date_range(start='2000-01-01', periods=N2)
        elif dtype == 'int64':
            level2 = range(N2)
        elif dtype == 'string':
            level2 = tm.makeStringIndex(N2)
        else:
            raise NotImplementedError
        self.left = MultiIndex.from_product([left_level1, level2])
        self.right = MultiIndex.from_product([right_level1, level2])

    def time_append(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        self.left.append(self.right)
from .pandas_vb_common import setup