import pandas as pd

class IndexCache:
    number = 1
    repeat = (3, 100, 20)
    params = [['CategoricalIndex', 'DatetimeIndex', 'Float64Index', 'IntervalIndex', 'Int64Index', 'MultiIndex', 'PeriodIndex', 'RangeIndex', 'TimedeltaIndex', 'UInt64Index']]
    param_names = ['index_type']

    def setup(self, index_type):
        if False:
            for i in range(10):
                print('nop')
        N = 10 ** 5
        if index_type == 'MultiIndex':
            self.idx = pd.MultiIndex.from_product([pd.date_range('1/1/2000', freq='min', periods=N // 2), ['a', 'b']])
        elif index_type == 'DatetimeIndex':
            self.idx = pd.date_range('1/1/2000', freq='min', periods=N)
        elif index_type == 'Int64Index':
            self.idx = pd.Index(range(N), dtype='int64')
        elif index_type == 'PeriodIndex':
            self.idx = pd.period_range('1/1/2000', freq='min', periods=N)
        elif index_type == 'RangeIndex':
            self.idx = pd.RangeIndex(start=0, stop=N)
        elif index_type == 'IntervalIndex':
            self.idx = pd.IntervalIndex.from_arrays(range(N), range(1, N + 1))
        elif index_type == 'TimedeltaIndex':
            self.idx = pd.TimedeltaIndex(range(N))
        elif index_type == 'Float64Index':
            self.idx = pd.Index(range(N), dtype='float64')
        elif index_type == 'UInt64Index':
            self.idx = pd.Index(range(N), dtype='uint64')
        elif index_type == 'CategoricalIndex':
            self.idx = pd.CategoricalIndex(range(N), range(N))
        else:
            raise ValueError
        assert len(self.idx) == N
        self.idx._cache = {}

    def time_values(self, index_type):
        if False:
            for i in range(10):
                print('nop')
        self.idx._values

    def time_shape(self, index_type):
        if False:
            print('Hello World!')
        self.idx.shape

    def time_is_monotonic_decreasing(self, index_type):
        if False:
            i = 10
            return i + 15
        self.idx.is_monotonic_decreasing

    def time_is_monotonic_increasing(self, index_type):
        if False:
            while True:
                i = 10
        self.idx.is_monotonic_increasing

    def time_is_unique(self, index_type):
        if False:
            i = 10
            return i + 15
        self.idx.is_unique

    def time_engine(self, index_type):
        if False:
            while True:
                i = 10
        self.idx._engine

    def time_inferred_type(self, index_type):
        if False:
            print('Hello World!')
        self.idx.inferred_type