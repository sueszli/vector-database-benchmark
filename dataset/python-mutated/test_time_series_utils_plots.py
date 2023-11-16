"""Module to test time_series plotting functionality
"""
from typing import List
import pandas as pd
import pytest
from plotly.subplots import make_subplots
from time_series_test_utils import _ALL_PLOTS
from pycaret.internal.plots.utils.time_series import ALLOWED_PLOT_DATA_TYPES, MULTIPLE_PLOT_TYPES_ALLOWED_AT_ONCE, _get_data_types_to_plot, _plot_fig_update, _reformat_dataframes_for_plots
pytestmark = [pytest.mark.filterwarnings('ignore::UserWarning'), pytest.mark.plotting]

@pytest.mark.parametrize('plot', _ALL_PLOTS)
def test_get_data_types_to_plot(plot):
    if False:
        return 10
    '_summary_'
    if plot is not None:
        returned_val = _get_data_types_to_plot(plot=plot)
        expected = [ALLOWED_PLOT_DATA_TYPES.get(plot)[0]]
        assert isinstance(returned_val, List)
        assert returned_val == expected
        data_types_requested = ALLOWED_PLOT_DATA_TYPES.get(plot)
        returned_val = _get_data_types_to_plot(plot=plot, data_types_requested=data_types_requested)
        assert isinstance(returned_val, List)
        accepts_multiple = MULTIPLE_PLOT_TYPES_ALLOWED_AT_ONCE.get(plot)
        if accepts_multiple:
            assert returned_val == data_types_requested
        else:
            assert returned_val == [data_types_requested[0]]
        with pytest.raises(ValueError) as errmsg:
            _ = _get_data_types_to_plot(plot=plot, data_types_requested='wrong')
        exceptionmsg = errmsg.value.args[0]
        assert 'No data to plot. Please check to make sure that you have requested an allowed data type for plot' in exceptionmsg

def test_reformat_dataframes_for_plots():
    if False:
        return 10
    'Tests for _reformat_dataframes_for_plots'
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    df2 = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    input_dfs = [df1, df2]
    labels_suffix = ['original', 'imputed']
    expected_cols = [['a (original)', 'a (imputed)'], ['b (original)', 'b (imputed)'], ['c (original)', 'c (imputed)']]
    output_dfs = _reformat_dataframes_for_plots(data=input_dfs, labels_suffix=labels_suffix)
    assert isinstance(output_dfs, List)
    for (item, expected_cols) in zip(output_dfs, expected_cols):
        assert isinstance(item, pd.DataFrame)
        assert item.columns.to_list() == expected_cols
    with pytest.raises(ValueError) as errmsg:
        labels_suffix = ['original']
        output_dfs = _reformat_dataframes_for_plots(data=input_dfs, labels_suffix=labels_suffix)
    exceptionmsg = errmsg.value.args[0]
    assert 'does not match the number of input dataframes' in exceptionmsg

def test_update_plot_config():
    if False:
        return 10
    'Tests for _plot_fig_update'
    title = 'main-title'
    subplot_title = 'subplot-title'
    fig_defaults = {'template': 'plotly', 'width': 10, 'height': 15}
    fig_kwargs = {}
    fig = make_subplots(rows=1, cols=1, row_heights=[0.33], subplot_titles=[subplot_title])
    fig = _plot_fig_update(fig, title, fig_defaults, fig_kwargs)
    assert fig.layout.annotations[0]['text'] == 'subplot-title'
    assert fig.layout.title.text == 'main-title'
    assert not fig.layout.showlegend
    assert fig.layout.width == 10
    assert fig.layout.height == 15
    fig = _plot_fig_update(fig, title, fig_defaults, fig_kwargs, show_legend=True)
    assert fig.layout.showlegend