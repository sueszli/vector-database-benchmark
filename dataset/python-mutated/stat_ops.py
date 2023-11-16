import numpy as np
import pandas as pd
ops = ['mean', 'sum', 'median', 'std', 'skew', 'kurt', 'prod', 'sem', 'var']

class FrameOps:
    params = [ops, ['float', 'int', 'Int64'], [0, 1, None]]
    param_names = ['op', 'dtype', 'axis']

    def setup(self, op, dtype, axis):
        if False:
            for i in range(10):
                print('nop')
        values = np.random.randn(100000, 4)
        if dtype == 'Int64':
            values = values.astype(int)
        df = pd.DataFrame(values).astype(dtype)
        self.df_func = getattr(df, op)

    def time_op(self, op, dtype, axis):
        if False:
            while True:
                i = 10
        self.df_func(axis=axis)

class FrameMixedDtypesOps:
    params = [ops, [0, 1, None]]
    param_names = ['op', 'axis']

    def setup(self, op, axis):
        if False:
            i = 10
            return i + 15
        if op in ('sum', 'skew', 'kurt', 'prod', 'sem', 'var') or (op, axis) in (('mean', 1), ('mean', None), ('median', 1), ('median', None), ('std', 1)):
            raise NotImplementedError
        N = 1000000
        df = pd.DataFrame({'f': np.random.normal(0.0, 1.0, N), 'i': np.random.randint(0, N, N), 'ts': pd.date_range(start='1/1/2000', periods=N, freq='h')})
        self.df_func = getattr(df, op)

    def time_op(self, op, axis):
        if False:
            return 10
        self.df_func(axis=axis)

class FrameMultiIndexOps:
    params = [ops]
    param_names = ['op']

    def setup(self, op):
        if False:
            print('Hello World!')
        levels = [np.arange(10), np.arange(100), np.arange(100)]
        codes = [np.arange(10).repeat(10000), np.tile(np.arange(100).repeat(100), 10), np.tile(np.tile(np.arange(100), 100), 10)]
        index = pd.MultiIndex(levels=levels, codes=codes)
        df = pd.DataFrame(np.random.randn(len(index), 4), index=index)
        self.df_func = getattr(df, op)

    def time_op(self, op):
        if False:
            while True:
                i = 10
        self.df_func()

class SeriesOps:
    params = [ops, ['float', 'int']]
    param_names = ['op', 'dtype']

    def setup(self, op, dtype):
        if False:
            print('Hello World!')
        s = pd.Series(np.random.randn(100000)).astype(dtype)
        self.s_func = getattr(s, op)

    def time_op(self, op, dtype):
        if False:
            return 10
        self.s_func()

class SeriesMultiIndexOps:
    params = [ops]
    param_names = ['op']

    def setup(self, op):
        if False:
            return 10
        levels = [np.arange(10), np.arange(100), np.arange(100)]
        codes = [np.arange(10).repeat(10000), np.tile(np.arange(100).repeat(100), 10), np.tile(np.tile(np.arange(100), 100), 10)]
        index = pd.MultiIndex(levels=levels, codes=codes)
        s = pd.Series(np.random.randn(len(index)), index=index)
        self.s_func = getattr(s, op)

    def time_op(self, op):
        if False:
            print('Hello World!')
        self.s_func()

class Rank:
    params = [['DataFrame', 'Series'], [True, False]]
    param_names = ['constructor', 'pct']

    def setup(self, constructor, pct):
        if False:
            print('Hello World!')
        values = np.random.randn(10 ** 5)
        self.data = getattr(pd, constructor)(values)

    def time_rank(self, constructor, pct):
        if False:
            for i in range(10):
                print('nop')
        self.data.rank(pct=pct)

    def time_average_old(self, constructor, pct):
        if False:
            return 10
        self.data.rank(pct=pct) / len(self.data)

class Correlation:
    params = [['spearman', 'kendall', 'pearson']]
    param_names = ['method']

    def setup(self, method):
        if False:
            for i in range(10):
                print('nop')
        self.df = pd.DataFrame(np.random.randn(500, 15))
        self.df2 = pd.DataFrame(np.random.randn(500, 15))
        self.df_wide = pd.DataFrame(np.random.randn(500, 100))
        self.df_wide_nans = self.df_wide.where(np.random.random((500, 100)) < 0.9)
        self.s = pd.Series(np.random.randn(500))
        self.s2 = pd.Series(np.random.randn(500))

    def time_corr(self, method):
        if False:
            return 10
        self.df.corr(method=method)

    def time_corr_wide(self, method):
        if False:
            for i in range(10):
                print('nop')
        self.df_wide.corr(method=method)

    def time_corr_wide_nans(self, method):
        if False:
            while True:
                i = 10
        self.df_wide_nans.corr(method=method)

    def peakmem_corr_wide(self, method):
        if False:
            print('Hello World!')
        self.df_wide.corr(method=method)

    def time_corr_series(self, method):
        if False:
            while True:
                i = 10
        self.s.corr(self.s2, method=method)

    def time_corrwith_cols(self, method):
        if False:
            print('Hello World!')
        self.df.corrwith(self.df2, method=method)

    def time_corrwith_rows(self, method):
        if False:
            i = 10
            return i + 15
        self.df.corrwith(self.df2, axis=1, method=method)

class Covariance:
    params = []
    param_names = []

    def setup(self):
        if False:
            return 10
        self.s = pd.Series(np.random.randn(100000))
        self.s2 = pd.Series(np.random.randn(100000))

    def time_cov_series(self):
        if False:
            for i in range(10):
                print('nop')
        self.s.cov(self.s2)
from .pandas_vb_common import setup