import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from seaborn._core.groupby import GroupBy
from seaborn._stats.aggregation import Agg, Est

class AggregationFixtures:

    @pytest.fixture
    def df(self, rng):
        if False:
            for i in range(10):
                print('nop')
        n = 30
        return pd.DataFrame(dict(x=rng.uniform(0, 7, n).round(), y=rng.normal(size=n), color=rng.choice(['a', 'b', 'c'], n), group=rng.choice(['x', 'y'], n)))

    def get_groupby(self, df, orient):
        if False:
            i = 10
            return i + 15
        other = {'x': 'y', 'y': 'x'}[orient]
        cols = [c for c in df if c != other]
        return GroupBy(cols)

class TestAgg(AggregationFixtures):

    def test_default(self, df):
        if False:
            print('Hello World!')
        ori = 'x'
        df = df[['x', 'y']]
        gb = self.get_groupby(df, ori)
        res = Agg()(df, gb, ori, {})
        expected = df.groupby('x', as_index=False)['y'].mean()
        assert_frame_equal(res, expected)

    def test_default_multi(self, df):
        if False:
            for i in range(10):
                print('nop')
        ori = 'x'
        gb = self.get_groupby(df, ori)
        res = Agg()(df, gb, ori, {})
        grp = ['x', 'color', 'group']
        index = pd.MultiIndex.from_product([sorted(df['x'].unique()), df['color'].unique(), df['group'].unique()], names=['x', 'color', 'group'])
        expected = df.groupby(grp).agg('mean').reindex(index=index).dropna().reset_index().reindex(columns=df.columns)
        assert_frame_equal(res, expected)

    @pytest.mark.parametrize('func', ['max', lambda x: float(len(x) % 2)])
    def test_func(self, df, func):
        if False:
            for i in range(10):
                print('nop')
        ori = 'x'
        df = df[['x', 'y']]
        gb = self.get_groupby(df, ori)
        res = Agg(func)(df, gb, ori, {})
        expected = df.groupby('x', as_index=False)['y'].agg(func)
        assert_frame_equal(res, expected)

class TestEst(AggregationFixtures):

    @pytest.mark.parametrize('func', [np.mean, 'mean'])
    def test_mean_sd(self, df, func):
        if False:
            print('Hello World!')
        ori = 'x'
        df = df[['x', 'y']]
        gb = self.get_groupby(df, ori)
        res = Est(func, 'sd')(df, gb, ori, {})
        grouped = df.groupby('x', as_index=False)['y']
        est = grouped.mean()
        err = grouped.std().fillna(0)
        expected = est.assign(ymin=est['y'] - err['y'], ymax=est['y'] + err['y'])
        assert_frame_equal(res, expected)

    def test_sd_single_obs(self):
        if False:
            while True:
                i = 10
        y = 1.5
        ori = 'x'
        df = pd.DataFrame([{'x': 'a', 'y': y}])
        gb = self.get_groupby(df, ori)
        res = Est('mean', 'sd')(df, gb, ori, {})
        expected = df.assign(ymin=y, ymax=y)
        assert_frame_equal(res, expected)

    def test_median_pi(self, df):
        if False:
            while True:
                i = 10
        ori = 'x'
        df = df[['x', 'y']]
        gb = self.get_groupby(df, ori)
        res = Est('median', ('pi', 100))(df, gb, ori, {})
        grouped = df.groupby('x', as_index=False)['y']
        est = grouped.median()
        expected = est.assign(ymin=grouped.min()['y'], ymax=grouped.max()['y'])
        assert_frame_equal(res, expected)

    def test_seed(self, df):
        if False:
            while True:
                i = 10
        ori = 'x'
        gb = self.get_groupby(df, ori)
        args = (df, gb, ori, {})
        res1 = Est('mean', 'ci', seed=99)(*args)
        res2 = Est('mean', 'ci', seed=99)(*args)
        assert_frame_equal(res1, res2)