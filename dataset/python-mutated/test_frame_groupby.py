""" Test cases for DataFrame.plot """
import pytest
from pandas import DataFrame
from pandas.tests.plotting.common import _check_visible
pytest.importorskip('matplotlib')

class TestDataFramePlotsGroupby:

    def _assert_ytickslabels_visibility(self, axes, expected):
        if False:
            print('Hello World!')
        for (ax, exp) in zip(axes, expected):
            _check_visible(ax.get_yticklabels(), visible=exp)

    def _assert_xtickslabels_visibility(self, axes, expected):
        if False:
            i = 10
            return i + 15
        for (ax, exp) in zip(axes, expected):
            _check_visible(ax.get_xticklabels(), visible=exp)

    @pytest.mark.parametrize('kwargs, expected', [({}, [True, False, True, False]), ({'sharey': True}, [True, False, True, False]), ({'sharey': False}, [True, True, True, True])])
    def test_groupby_boxplot_sharey(self, kwargs, expected):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame({'a': [-1.43, -0.15, -3.7, -1.43, -0.14], 'b': [0.56, 0.84, 0.29, 0.56, 0.85], 'c': [0, 1, 2, 3, 1]}, index=[0, 1, 2, 3, 4])
        axes = df.groupby('c').boxplot(**kwargs)
        self._assert_ytickslabels_visibility(axes, expected)

    @pytest.mark.parametrize('kwargs, expected', [({}, [True, True, True, True]), ({'sharex': False}, [True, True, True, True]), ({'sharex': True}, [False, False, True, True])])
    def test_groupby_boxplot_sharex(self, kwargs, expected):
        if False:
            i = 10
            return i + 15
        df = DataFrame({'a': [-1.43, -0.15, -3.7, -1.43, -0.14], 'b': [0.56, 0.84, 0.29, 0.56, 0.85], 'c': [0, 1, 2, 3, 1]}, index=[0, 1, 2, 3, 4])
        axes = df.groupby('c').boxplot(**kwargs)
        self._assert_xtickslabels_visibility(axes, expected)