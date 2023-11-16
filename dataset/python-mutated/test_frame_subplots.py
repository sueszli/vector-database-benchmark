""" Test cases for DataFrame.plot """
import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import DataFrame, Series, date_range
import pandas._testing as tm
from pandas.tests.plotting.common import _check_axes_shape, _check_box_return_type, _check_legend_labels, _check_ticks_props, _check_visible, _flatten_visible
from pandas.io.formats.printing import pprint_thing
mpl = pytest.importorskip('matplotlib')
plt = pytest.importorskip('matplotlib.pyplot')

class TestDataFramePlotsSubplots:

    @pytest.mark.slow
    @pytest.mark.parametrize('kind', ['bar', 'barh', 'line', 'area'])
    def test_subplots(self, kind):
        if False:
            print('Hello World!')
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        axes = df.plot(kind=kind, subplots=True, sharex=True, legend=True)
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))
        assert axes.shape == (3,)
        for (ax, column) in zip(axes, df.columns):
            _check_legend_labels(ax, labels=[pprint_thing(column)])
        for ax in axes[:-2]:
            _check_visible(ax.xaxis)
            _check_visible(ax.get_xticklabels(), visible=False)
            if kind != 'bar':
                _check_visible(ax.get_xticklabels(minor=True), visible=False)
            _check_visible(ax.xaxis.get_label(), visible=False)
            _check_visible(ax.get_yticklabels())
        _check_visible(axes[-1].xaxis)
        _check_visible(axes[-1].get_xticklabels())
        _check_visible(axes[-1].get_xticklabels(minor=True))
        _check_visible(axes[-1].xaxis.get_label())
        _check_visible(axes[-1].get_yticklabels())

    @pytest.mark.slow
    @pytest.mark.parametrize('kind', ['bar', 'barh', 'line', 'area'])
    def test_subplots_no_share_x(self, kind):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        axes = df.plot(kind=kind, subplots=True, sharex=False)
        for ax in axes:
            _check_visible(ax.xaxis)
            _check_visible(ax.get_xticklabels())
            _check_visible(ax.get_xticklabels(minor=True))
            _check_visible(ax.xaxis.get_label())
            _check_visible(ax.get_yticklabels())

    @pytest.mark.slow
    @pytest.mark.parametrize('kind', ['bar', 'barh', 'line', 'area'])
    def test_subplots_no_legend(self, kind):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        axes = df.plot(kind=kind, subplots=True, legend=False)
        for ax in axes:
            assert ax.get_legend() is None

    @pytest.mark.parametrize('kind', ['line', 'area'])
    def test_subplots_timeseries(self, kind):
        if False:
            for i in range(10):
                print('nop')
        idx = date_range(start='2014-07-01', freq='ME', periods=10)
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)
        axes = df.plot(kind=kind, subplots=True, sharex=True)
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))
        for ax in axes[:-2]:
            _check_visible(ax.xaxis)
            _check_visible(ax.get_xticklabels(), visible=False)
            _check_visible(ax.get_xticklabels(minor=True), visible=False)
            _check_visible(ax.xaxis.get_label(), visible=False)
            _check_visible(ax.get_yticklabels())
        _check_visible(axes[-1].xaxis)
        _check_visible(axes[-1].get_xticklabels())
        _check_visible(axes[-1].get_xticklabels(minor=True))
        _check_visible(axes[-1].xaxis.get_label())
        _check_visible(axes[-1].get_yticklabels())
        _check_ticks_props(axes, xrot=0)

    @pytest.mark.parametrize('kind', ['line', 'area'])
    def test_subplots_timeseries_rot(self, kind):
        if False:
            while True:
                i = 10
        idx = date_range(start='2014-07-01', freq='ME', periods=10)
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)
        axes = df.plot(kind=kind, subplots=True, sharex=False, rot=45, fontsize=7)
        for ax in axes:
            _check_visible(ax.xaxis)
            _check_visible(ax.get_xticklabels())
            _check_visible(ax.get_xticklabels(minor=True))
            _check_visible(ax.xaxis.get_label())
            _check_visible(ax.get_yticklabels())
            _check_ticks_props(ax, xlabelsize=7, xrot=45, ylabelsize=7)

    @pytest.mark.parametrize('col', ['numeric', 'timedelta', 'datetime_no_tz', 'datetime_all_tz'])
    def test_subplots_timeseries_y_axis(self, col):
        if False:
            i = 10
            return i + 15
        data = {'numeric': np.array([1, 2, 5]), 'timedelta': [pd.Timedelta(-10, unit='s'), pd.Timedelta(10, unit='m'), pd.Timedelta(10, unit='h')], 'datetime_no_tz': [pd.to_datetime('2017-08-01 00:00:00'), pd.to_datetime('2017-08-01 02:00:00'), pd.to_datetime('2017-08-02 00:00:00')], 'datetime_all_tz': [pd.to_datetime('2017-08-01 00:00:00', utc=True), pd.to_datetime('2017-08-01 02:00:00', utc=True), pd.to_datetime('2017-08-02 00:00:00', utc=True)], 'text': ['This', 'should', 'fail']}
        testdata = DataFrame(data)
        ax = testdata.plot(y=col)
        result = ax.get_lines()[0].get_data()[1]
        expected = testdata[col].values
        assert (result == expected).all()

    def test_subplots_timeseries_y_text_error(self):
        if False:
            print('Hello World!')
        data = {'numeric': np.array([1, 2, 5]), 'text': ['This', 'should', 'fail']}
        testdata = DataFrame(data)
        msg = 'no numeric data to plot'
        with pytest.raises(TypeError, match=msg):
            testdata.plot(y='text')

    @pytest.mark.xfail(reason='not support for period, categorical, datetime_mixed_tz')
    def test_subplots_timeseries_y_axis_not_supported(self):
        if False:
            return 10
        "\n        This test will fail for:\n            period:\n                since period isn't yet implemented in ``select_dtypes``\n                and because it will need a custom value converter +\n                tick formatter (as was done for x-axis plots)\n\n            categorical:\n                 because it will need a custom value converter +\n                 tick formatter (also doesn't work for x-axis, as of now)\n\n            datetime_mixed_tz:\n                because of the way how pandas handles ``Series`` of\n                ``datetime`` objects with different timezone,\n                generally converting ``datetime`` objects in a tz-aware\n                form could help with this problem\n        "
        data = {'numeric': np.array([1, 2, 5]), 'period': [pd.Period('2017-08-01 00:00:00', freq='H'), pd.Period('2017-08-01 02:00', freq='H'), pd.Period('2017-08-02 00:00:00', freq='H')], 'categorical': pd.Categorical(['c', 'b', 'a'], categories=['a', 'b', 'c'], ordered=False), 'datetime_mixed_tz': [pd.to_datetime('2017-08-01 00:00:00', utc=True), pd.to_datetime('2017-08-01 02:00:00'), pd.to_datetime('2017-08-02 00:00:00')]}
        testdata = DataFrame(data)
        ax_period = testdata.plot(x='numeric', y='period')
        assert (ax_period.get_lines()[0].get_data()[1] == testdata['period'].values).all()
        ax_categorical = testdata.plot(x='numeric', y='categorical')
        assert (ax_categorical.get_lines()[0].get_data()[1] == testdata['categorical'].values).all()
        ax_datetime_mixed_tz = testdata.plot(x='numeric', y='datetime_mixed_tz')
        assert (ax_datetime_mixed_tz.get_lines()[0].get_data()[1] == testdata['datetime_mixed_tz'].values).all()

    @pytest.mark.parametrize('layout, exp_layout', [[(2, 2), (2, 2)], [(-1, 2), (2, 2)], [(2, -1), (2, 2)], [(1, 4), (1, 4)], [(-1, 4), (1, 4)], [(4, -1), (4, 1)]])
    def test_subplots_layout_multi_column(self, layout, exp_layout):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        axes = df.plot(subplots=True, layout=layout)
        _check_axes_shape(axes, axes_num=3, layout=exp_layout)
        assert axes.shape == exp_layout

    def test_subplots_layout_multi_column_error(self):
        if False:
            i = 10
            return i + 15
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        msg = 'Layout of 1x1 must be larger than required size 3'
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, layout=(1, 1))
        msg = 'At least one dimension of layout must be positive'
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, layout=(-1, -1))

    @pytest.mark.parametrize('kwargs, expected_axes_num, expected_layout, expected_shape', [({}, 1, (1, 1), (1,)), ({'layout': (3, 3)}, 1, (3, 3), (3, 3))])
    def test_subplots_layout_single_column(self, kwargs, expected_axes_num, expected_layout, expected_shape):
        if False:
            i = 10
            return i + 15
        df = DataFrame(np.random.default_rng(2).random((10, 1)), index=list(string.ascii_letters[:10]))
        axes = df.plot(subplots=True, **kwargs)
        _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
        assert axes.shape == expected_shape

    @pytest.mark.slow
    @pytest.mark.parametrize('idx', [range(5), date_range('1/1/2000', periods=5)])
    def test_subplots_warnings(self, idx):
        if False:
            while True:
                i = 10
        with tm.assert_produces_warning(None):
            df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), index=idx)
            df.plot(subplots=True, layout=(3, 2))

    def test_subplots_multiple_axes(self):
        if False:
            for i in range(10):
                print('nop')
        (fig, axes) = mpl.pyplot.subplots(2, 3)
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        returned = df.plot(subplots=True, ax=axes[0], sharex=False, sharey=False)
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        assert returned.shape == (3,)
        assert returned[0].figure is fig
        returned = df.plot(subplots=True, ax=axes[1], sharex=False, sharey=False)
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        assert returned.shape == (3,)
        assert returned[0].figure is fig
        _check_axes_shape(axes, axes_num=6, layout=(2, 3))

    def test_subplots_multiple_axes_error(self):
        if False:
            print('Hello World!')
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
        msg = 'The number of passed axes must be 3, the same as the output plot'
        (_, axes) = mpl.pyplot.subplots(2, 3)
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, ax=axes)

    @pytest.mark.parametrize('layout, exp_layout', [[(2, 1), (2, 2)], [(2, -1), (2, 2)], [(-1, 2), (2, 2)]])
    def test_subplots_multiple_axes_2_dim(self, layout, exp_layout):
        if False:
            while True:
                i = 10
        (_, axes) = mpl.pyplot.subplots(2, 2)
        df = DataFrame(np.random.default_rng(2).random((10, 4)), index=list(string.ascii_letters[:10]))
        with tm.assert_produces_warning(UserWarning):
            returned = df.plot(subplots=True, ax=axes, layout=layout, sharex=False, sharey=False)
            _check_axes_shape(returned, axes_num=4, layout=exp_layout)
            assert returned.shape == (4,)

    def test_subplots_multiple_axes_single_col(self):
        if False:
            print('Hello World!')
        (_, axes) = mpl.pyplot.subplots(1, 1)
        df = DataFrame(np.random.default_rng(2).random((10, 1)), index=list(string.ascii_letters[:10]))
        axes = df.plot(subplots=True, ax=[axes], sharex=False, sharey=False)
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))
        assert axes.shape == (1,)

    def test_subplots_ts_share_axes(self):
        if False:
            while True:
                i = 10
        (_, axes) = mpl.pyplot.subplots(3, 3, sharex=True, sharey=True)
        mpl.pyplot.subplots_adjust(left=0.05, right=0.95, hspace=0.3, wspace=0.3)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 9)), index=date_range(start='2014-07-01', freq='ME', periods=10))
        for (i, ax) in enumerate(axes.ravel()):
            df[i].plot(ax=ax, fontsize=5)
        for ax in axes[0:-1].ravel():
            _check_visible(ax.get_xticklabels(), visible=False)
        for ax in axes[-1].ravel():
            _check_visible(ax.get_xticklabels(), visible=True)
        for ax in axes[[0, 1, 2], [0]].ravel():
            _check_visible(ax.get_yticklabels(), visible=True)
        for ax in axes[[0, 1, 2], [1]].ravel():
            _check_visible(ax.get_yticklabels(), visible=False)
        for ax in axes[[0, 1, 2], [2]].ravel():
            _check_visible(ax.get_yticklabels(), visible=False)

    def test_subplots_sharex_axes_existing_axes(self):
        if False:
            i = 10
            return i + 15
        d = {'A': [1.0, 2.0, 3.0, 4.0], 'B': [4.0, 3.0, 2.0, 1.0], 'C': [5, 1, 3, 4]}
        df = DataFrame(d, index=date_range('2014 10 11', '2014 10 14'))
        axes = df[['A', 'B']].plot(subplots=True)
        df['C'].plot(ax=axes[0], secondary_y=True)
        _check_visible(axes[0].get_xticklabels(), visible=False)
        _check_visible(axes[1].get_xticklabels(), visible=True)
        for ax in axes.ravel():
            _check_visible(ax.get_yticklabels(), visible=True)

    def test_subplots_dup_columns(self):
        if False:
            while True:
                i = 10
        df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=list('aaaaa'))
        axes = df.plot(subplots=True)
        for ax in axes:
            _check_legend_labels(ax, labels=['a'])
            assert len(ax.lines) == 1

    def test_subplots_dup_columns_secondary_y(self):
        if False:
            i = 10
            return i + 15
        df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=list('aaaaa'))
        axes = df.plot(subplots=True, secondary_y='a')
        for ax in axes:
            _check_legend_labels(ax, labels=['a'])
            assert len(ax.lines) == 1

    def test_subplots_dup_columns_secondary_y_no_subplot(self):
        if False:
            while True:
                i = 10
        df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=list('aaaaa'))
        ax = df.plot(secondary_y='a')
        _check_legend_labels(ax, labels=['a (right)'] * 5)
        assert len(ax.lines) == 0
        assert len(ax.right_ax.lines) == 5

    @pytest.mark.xfail(np_version_gte1p24 and is_platform_linux(), reason='Weird rounding problems', strict=False)
    def test_bar_log_no_subplots(self):
        if False:
            while True:
                i = 10
        expected = np.array([0.1, 1.0, 10.0, 100])
        df = DataFrame({'A': [3] * 5, 'B': list(range(1, 6))}, index=range(5))
        ax = df.plot.bar(grid=True, log=True)
        tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), expected)

    @pytest.mark.xfail(np_version_gte1p24 and is_platform_linux(), reason='Weird rounding problems', strict=False)
    def test_bar_log_subplots(self):
        if False:
            print('Hello World!')
        expected = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
        ax = DataFrame([Series([200, 300]), Series([300, 500])]).plot.bar(log=True, subplots=True)
        tm.assert_numpy_array_equal(ax[0].yaxis.get_ticklocs(), expected)
        tm.assert_numpy_array_equal(ax[1].yaxis.get_ticklocs(), expected)

    def test_boxplot_subplots_return_type_default(self, hist_df):
        if False:
            print('Hello World!')
        df = hist_df
        result = df.plot.box(subplots=True)
        assert isinstance(result, Series)
        _check_box_return_type(result, None, expected_keys=['height', 'weight', 'category'])

    @pytest.mark.parametrize('rt', ['dict', 'axes', 'both'])
    def test_boxplot_subplots_return_type(self, hist_df, rt):
        if False:
            return 10
        df = hist_df
        returned = df.plot.box(return_type=rt, subplots=True)
        _check_box_return_type(returned, rt, expected_keys=['height', 'weight', 'category'], check_ax_title=False)

    def test_df_subplots_patterns_minorticks(self):
        if False:
            print('Hello World!')
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=date_range('1/1/2000', periods=10), columns=list('AB'))
        (_, axes) = plt.subplots(2, 1, sharex=True)
        axes = df.plot(subplots=True, ax=axes)
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
        _check_visible(axes[0].get_xticklabels(), visible=False)
        _check_visible(axes[0].get_xticklabels(minor=True), visible=False)
        _check_visible(axes[1].get_xticklabels(), visible=True)
        _check_visible(axes[1].get_xticklabels(minor=True), visible=True)

    def test_df_subplots_patterns_minorticks_1st_ax_hidden(self):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=date_range('1/1/2000', periods=10), columns=list('AB'))
        (_, axes) = plt.subplots(2, 1)
        with tm.assert_produces_warning(UserWarning):
            axes = df.plot(subplots=True, ax=axes, sharex=True)
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
        _check_visible(axes[0].get_xticklabels(), visible=False)
        _check_visible(axes[0].get_xticklabels(minor=True), visible=False)
        _check_visible(axes[1].get_xticklabels(), visible=True)
        _check_visible(axes[1].get_xticklabels(minor=True), visible=True)

    def test_df_subplots_patterns_minorticks_not_shared(self):
        if False:
            i = 10
            return i + 15
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=date_range('1/1/2000', periods=10), columns=list('AB'))
        (_, axes) = plt.subplots(2, 1)
        axes = df.plot(subplots=True, ax=axes)
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)

    def test_subplots_sharex_false(self):
        if False:
            return 10
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        df.iloc[5:, 1] = np.nan
        df.iloc[:5, 0] = np.nan
        (_, axs) = mpl.pyplot.subplots(2, 1)
        df.plot.line(ax=axs, subplots=True, sharex=False)
        expected_ax1 = np.arange(4.5, 10, 0.5)
        expected_ax2 = np.arange(-0.5, 5, 0.5)
        tm.assert_numpy_array_equal(axs[0].get_xticks(), expected_ax1)
        tm.assert_numpy_array_equal(axs[1].get_xticks(), expected_ax2)

    def test_subplots_constrained_layout(self):
        if False:
            for i in range(10):
                print('nop')
        idx = date_range(start='now', periods=10)
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)
        kwargs = {}
        if hasattr(mpl.pyplot.Figure, 'get_constrained_layout'):
            kwargs['constrained_layout'] = True
        (_, axes) = mpl.pyplot.subplots(2, **kwargs)
        with tm.assert_produces_warning(None):
            df.plot(ax=axes[0])
            with tm.ensure_clean(return_filelike=True) as path:
                mpl.pyplot.savefig(path)

    @pytest.mark.parametrize('index_name, old_label, new_label', [(None, '', 'new'), ('old', 'old', 'new'), (None, '', ''), (None, '', 1), (None, '', [1, 2])])
    @pytest.mark.parametrize('kind', ['line', 'area', 'bar'])
    def test_xlabel_ylabel_dataframe_subplots(self, kind, index_name, old_label, new_label):
        if False:
            i = 10
            return i + 15
        df = DataFrame([[1, 2], [2, 5]], columns=['Type A', 'Type B'])
        df.index.name = index_name
        axes = df.plot(kind=kind, subplots=True)
        assert all((ax.get_ylabel() == '' for ax in axes))
        assert all((ax.get_xlabel() == old_label for ax in axes))
        axes = df.plot(kind=kind, ylabel=new_label, xlabel=new_label, subplots=True)
        assert all((ax.get_ylabel() == str(new_label) for ax in axes))
        assert all((ax.get_xlabel() == str(new_label) for ax in axes))

    @pytest.mark.parametrize('kwargs', [{'kind': 'bar', 'stacked': True}, {'kind': 'bar', 'stacked': True, 'width': 0.9}, {'kind': 'barh', 'stacked': True}, {'kind': 'barh', 'stacked': True, 'width': 0.9}, {'kind': 'bar', 'stacked': False}, {'kind': 'bar', 'stacked': False, 'width': 0.9}, {'kind': 'barh', 'stacked': False}, {'kind': 'barh', 'stacked': False, 'width': 0.9}, {'kind': 'bar', 'subplots': True}, {'kind': 'bar', 'subplots': True, 'width': 0.9}, {'kind': 'barh', 'subplots': True}, {'kind': 'barh', 'subplots': True, 'width': 0.9}, {'kind': 'bar', 'stacked': True, 'align': 'edge'}, {'kind': 'bar', 'stacked': True, 'width': 0.9, 'align': 'edge'}, {'kind': 'barh', 'stacked': True, 'align': 'edge'}, {'kind': 'barh', 'stacked': True, 'width': 0.9, 'align': 'edge'}, {'kind': 'bar', 'stacked': False, 'align': 'edge'}, {'kind': 'bar', 'stacked': False, 'width': 0.9, 'align': 'edge'}, {'kind': 'barh', 'stacked': False, 'align': 'edge'}, {'kind': 'barh', 'stacked': False, 'width': 0.9, 'align': 'edge'}, {'kind': 'bar', 'subplots': True, 'align': 'edge'}, {'kind': 'bar', 'subplots': True, 'width': 0.9, 'align': 'edge'}, {'kind': 'barh', 'subplots': True, 'align': 'edge'}, {'kind': 'barh', 'subplots': True, 'width': 0.9, 'align': 'edge'}])
    def test_bar_align_multiple_columns(self, kwargs):
        if False:
            while True:
                i = 10
        df = DataFrame({'A': [3] * 5, 'B': list(range(5))}, index=range(5))
        self._check_bar_alignment(df, **kwargs)

    @pytest.mark.parametrize('kwargs', [{'kind': 'bar', 'stacked': False}, {'kind': 'bar', 'stacked': True}, {'kind': 'barh', 'stacked': False}, {'kind': 'barh', 'stacked': True}, {'kind': 'bar', 'subplots': True}, {'kind': 'barh', 'subplots': True}])
    def test_bar_align_single_column(self, kwargs):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame(np.random.default_rng(2).standard_normal(5))
        self._check_bar_alignment(df, **kwargs)

    @pytest.mark.parametrize('kwargs', [{'kind': 'bar', 'stacked': False}, {'kind': 'bar', 'stacked': True}, {'kind': 'barh', 'stacked': False}, {'kind': 'barh', 'stacked': True}, {'kind': 'bar', 'subplots': True}, {'kind': 'barh', 'subplots': True}])
    def test_bar_barwidth_position(self, kwargs):
        if False:
            i = 10
            return i + 15
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        self._check_bar_alignment(df, width=0.9, position=0.2, **kwargs)

    @pytest.mark.parametrize('w', [1, 1.0])
    def test_bar_barwidth_position_int(self, w):
        if False:
            while True:
                i = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar(stacked=True, width=w)
        ticks = ax.xaxis.get_ticklocs()
        tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4]))
        assert ax.get_xlim() == (-0.75, 4.75)
        assert ax.patches[0].get_x() == -0.5
        assert ax.patches[-1].get_x() == 3.5

    @pytest.mark.parametrize('kind, kwargs', [['bar', {'stacked': True}], ['barh', {'stacked': False}], ['barh', {'stacked': True}], ['bar', {'subplots': True}], ['barh', {'subplots': True}]])
    def test_bar_barwidth_position_int_width_1(self, kind, kwargs):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        self._check_bar_alignment(df, kind=kind, width=1, **kwargs)

    def _check_bar_alignment(self, df, kind='bar', stacked=False, subplots=False, align='center', width=0.5, position=0.5):
        if False:
            return 10
        axes = df.plot(kind=kind, stacked=stacked, subplots=subplots, align=align, width=width, position=position, grid=True)
        axes = _flatten_visible(axes)
        for ax in axes:
            if kind == 'bar':
                axis = ax.xaxis
                (ax_min, ax_max) = ax.get_xlim()
                min_edge = min((p.get_x() for p in ax.patches))
                max_edge = max((p.get_x() + p.get_width() for p in ax.patches))
            elif kind == 'barh':
                axis = ax.yaxis
                (ax_min, ax_max) = ax.get_ylim()
                min_edge = min((p.get_y() for p in ax.patches))
                max_edge = max((p.get_y() + p.get_height() for p in ax.patches))
            else:
                raise ValueError
            tm.assert_almost_equal(ax_min, min_edge - 0.25)
            tm.assert_almost_equal(ax_max, max_edge + 0.25)
            p = ax.patches[0]
            if kind == 'bar' and (stacked is True or subplots is True):
                edge = p.get_x()
                center = edge + p.get_width() * position
            elif kind == 'bar' and stacked is False:
                center = p.get_x() + p.get_width() * len(df.columns) * position
                edge = p.get_x()
            elif kind == 'barh' and (stacked is True or subplots is True):
                center = p.get_y() + p.get_height() * position
                edge = p.get_y()
            elif kind == 'barh' and stacked is False:
                center = p.get_y() + p.get_height() * len(df.columns) * position
                edge = p.get_y()
            else:
                raise ValueError
            assert (axis.get_ticklocs() == np.arange(len(df))).all()
            if align == 'center':
                tm.assert_almost_equal(axis.get_ticklocs()[0], center)
            elif align == 'edge':
                tm.assert_almost_equal(axis.get_ticklocs()[0], edge)
            else:
                raise ValueError
        return axes