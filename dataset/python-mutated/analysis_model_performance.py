from functools import partial
import pandas as pd
import plotly.graph_objs as go
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from typing import Sequence
from qlib.typehint import Literal
from ..graph import ScatterGraph, SubplotsGraph, BarGraph, HeatmapGraph
from ..utils import guess_plotly_rangebreaks

def _group_return(pred_label: pd.DataFrame=None, reverse: bool=False, N: int=5, **kwargs) -> tuple:
    if False:
        return 10
    '\n\n    :param pred_label:\n    :param reverse:\n    :param N:\n    :return:\n    '
    if reverse:
        pred_label['score'] *= -1
    pred_label = pred_label.sort_values('score', ascending=False)
    pred_label_drop = pred_label.dropna(subset=['score'])
    t_df = pd.DataFrame({'Group%d' % (i + 1): pred_label_drop.groupby(level='datetime')['label'].apply(lambda x: x[len(x) // N * i:len(x) // N * (i + 1)].mean()) for i in range(N)})
    t_df.index = pd.to_datetime(t_df.index)
    t_df['long-short'] = t_df['Group1'] - t_df['Group%d' % N]
    t_df['long-average'] = t_df['Group1'] - pred_label.groupby(level='datetime')['label'].mean()
    t_df = t_df.dropna(how='all')
    group_scatter_figure = ScatterGraph(t_df.cumsum(), layout=dict(title='Cumulative Return', xaxis=dict(tickangle=45, rangebreaks=kwargs.get('rangebreaks', guess_plotly_rangebreaks(t_df.index))))).figure
    t_df = t_df.loc[:, ['long-short', 'long-average']]
    _bin_size = float(((t_df.max() - t_df.min()) / 20).min())
    group_hist_figure = SubplotsGraph(t_df, kind_map=dict(kind='DistplotGraph', kwargs=dict(bin_size=_bin_size)), subplots_kwargs=dict(rows=1, cols=2, print_grid=False, subplot_titles=['long-short', 'long-average'])).figure
    return (group_scatter_figure, group_hist_figure)

def _plot_qq(data: pd.Series=None, dist=stats.norm) -> go.Figure:
    if False:
        print('Hello World!')
    '\n\n    :param data:\n    :param dist:\n    :return:\n    '
    _plt_fig = sm.qqplot(data.dropna(), dist=dist, fit=True, line='45')
    plt.close(_plt_fig)
    qqplot_data = _plt_fig.gca().lines
    fig = go.Figure()
    fig.add_trace({'type': 'scatter', 'x': qqplot_data[0].get_xdata(), 'y': qqplot_data[0].get_ydata(), 'mode': 'markers', 'marker': {'color': '#19d3f3'}})
    fig.add_trace({'type': 'scatter', 'x': qqplot_data[1].get_xdata(), 'y': qqplot_data[1].get_ydata(), 'mode': 'lines', 'line': {'color': '#636efa'}})
    del qqplot_data
    return fig

def _pred_ic(pred_label: pd.DataFrame=None, methods: Sequence[Literal['IC', 'Rank IC']]=('IC', 'Rank IC'), **kwargs) -> tuple:
    if False:
        i = 10
        return i + 15
    '\n\n    :param pred_label: pd.DataFrame\n    must contain one column of realized return with name `label` and one column of predicted score names `score`.\n    :param methods: Sequence[Literal["IC", "Rank IC"]]\n    IC series to plot.\n    IC is sectional pearson correlation between label and score\n    Rank IC is the spearman correlation between label and score\n    For the Monthly IC, IC histogram, IC Q-Q plot.  Only the first type of IC will be plotted.\n    :return:\n    '
    _methods_mapping = {'IC': 'pearson', 'Rank IC': 'spearman'}

    def _corr_series(x, method):
        if False:
            i = 10
            return i + 15
        return x['label'].corr(x['score'], method=method)
    ic_df = pd.concat([pred_label.groupby(level='datetime').apply(partial(_corr_series, method=_methods_mapping[m])).rename(m) for m in methods], axis=1)
    _ic = ic_df.iloc(axis=1)[0]
    _index = _ic.index.get_level_values(0).astype('str').str.replace('-', '').str.slice(0, 6)
    _monthly_ic = _ic.groupby(_index).mean()
    _monthly_ic.index = pd.MultiIndex.from_arrays([_monthly_ic.index.str.slice(0, 4), _monthly_ic.index.str.slice(4, 6)], names=['year', 'month'])
    _month_list = pd.date_range(start=pd.Timestamp(f'{_index.min()[:4]}0101'), end=pd.Timestamp(f'{_index.max()[:4]}1231'), freq='1M')
    _years = []
    _month = []
    for _date in _month_list:
        _date = _date.strftime('%Y%m%d')
        _years.append(_date[:4])
        _month.append(_date[4:6])
    fill_index = pd.MultiIndex.from_arrays([_years, _month], names=['year', 'month'])
    _monthly_ic = _monthly_ic.reindex(fill_index)
    ic_bar_figure = ic_figure(ic_df, kwargs.get('show_nature_day', False))
    ic_heatmap_figure = HeatmapGraph(_monthly_ic.unstack(), layout=dict(title='Monthly IC', xaxis=dict(dtick=1), yaxis=dict(tickformat='04d', dtick=1)), graph_kwargs=dict(xtype='array', ytype='array')).figure
    dist = stats.norm
    _qqplot_fig = _plot_qq(_ic, dist)
    if isinstance(dist, stats.norm.__class__):
        dist_name = 'Normal'
    else:
        dist_name = 'Unknown'
    _ic_df = _ic.to_frame('IC')
    _bin_size = ((_ic_df.max() - _ic_df.min()) / 20).min()
    _sub_graph_data = [('IC', dict(row=1, col=1, name='', kind='DistplotGraph', graph_kwargs=dict(bin_size=_bin_size))), (_qqplot_fig, dict(row=1, col=2))]
    ic_hist_figure = SubplotsGraph(_ic_df.dropna(), kind_map=dict(kind='HistogramGraph', kwargs=dict()), subplots_kwargs=dict(rows=1, cols=2, print_grid=False, subplot_titles=['IC', 'IC %s Dist. Q-Q' % dist_name]), sub_graph_data=_sub_graph_data, layout=dict(yaxis2=dict(title='Observed Quantile'), xaxis2=dict(title=f'{dist_name} Distribution Quantile'))).figure
    return (ic_bar_figure, ic_heatmap_figure, ic_hist_figure)

def _pred_autocorr(pred_label: pd.DataFrame, lag=1, **kwargs) -> tuple:
    if False:
        print('Hello World!')
    pred = pred_label.copy()
    pred['score_last'] = pred.groupby(level='instrument')['score'].shift(lag)
    ac = pred.groupby(level='datetime').apply(lambda x: x['score'].rank(pct=True).corr(x['score_last'].rank(pct=True)))
    _df = ac.to_frame('value')
    ac_figure = ScatterGraph(_df, layout=dict(title='Auto Correlation', xaxis=dict(tickangle=45, rangebreaks=kwargs.get('rangebreaks', guess_plotly_rangebreaks(_df.index))))).figure
    return (ac_figure,)

def _pred_turnover(pred_label: pd.DataFrame, N=5, lag=1, **kwargs) -> tuple:
    if False:
        print('Hello World!')
    pred = pred_label.copy()
    pred['score_last'] = pred.groupby(level='instrument')['score'].shift(lag)
    top = pred.groupby(level='datetime').apply(lambda x: 1 - x.nlargest(len(x) // N, columns='score').index.isin(x.nlargest(len(x) // N, columns='score_last').index).sum() / (len(x) // N))
    bottom = pred.groupby(level='datetime').apply(lambda x: 1 - x.nsmallest(len(x) // N, columns='score').index.isin(x.nsmallest(len(x) // N, columns='score_last').index).sum() / (len(x) // N))
    r_df = pd.DataFrame({'Top': top, 'Bottom': bottom})
    turnover_figure = ScatterGraph(r_df, layout=dict(title='Top-Bottom Turnover', xaxis=dict(tickangle=45, rangebreaks=kwargs.get('rangebreaks', guess_plotly_rangebreaks(r_df.index))))).figure
    return (turnover_figure,)

def ic_figure(ic_df: pd.DataFrame, show_nature_day=True, **kwargs) -> go.Figure:
    if False:
        return 10
    'IC figure\n\n    :param ic_df: ic DataFrame\n    :param show_nature_day: whether to display the abscissa of non-trading day\n    :param \\*\\*kwargs: contains some parameters to control plot style in plotly. Currently, supports\n       - `rangebreaks`: https://plotly.com/python/time-series/#Hiding-Weekends-and-Holidays\n    :return: plotly.graph_objs.Figure\n    '
    if show_nature_day:
        date_index = pd.date_range(ic_df.index.min(), ic_df.index.max())
        ic_df = ic_df.reindex(date_index)
    ic_bar_figure = BarGraph(ic_df, layout=dict(title='Information Coefficient (IC)', xaxis=dict(tickangle=45, rangebreaks=kwargs.get('rangebreaks', guess_plotly_rangebreaks(ic_df.index))))).figure
    return ic_bar_figure

def model_performance_graph(pred_label: pd.DataFrame, lag: int=1, N: int=5, reverse=False, rank=False, graph_names: list=['group_return', 'pred_ic', 'pred_autocorr'], show_notebook: bool=True, show_nature_day: bool=False, **kwargs) -> [list, tuple]:
    if False:
        return 10
    'Model performance\n\n    :param pred_label: index is **pd.MultiIndex**, index name is **[instrument, datetime]**; columns names is **[score, label]**.\n           It is usually same as the label of model training(e.g. "Ref($close, -2)/Ref($close, -1) - 1").\n\n\n            .. code-block:: python\n\n                instrument  datetime        score       label\n                SH600004    2017-12-11  -0.013502       -0.013502\n                                2017-12-12  -0.072367       -0.072367\n                                2017-12-13  -0.068605       -0.068605\n                                2017-12-14  0.012440        0.012440\n                                2017-12-15  -0.102778       -0.102778\n\n\n    :param lag: `pred.groupby(level=\'instrument\')[\'score\'].shift(lag)`. It will be only used in the auto-correlation computing.\n    :param N: group number, default 5.\n    :param reverse: if `True`, `pred[\'score\'] *= -1`.\n    :param rank: if **True**, calculate rank ic.\n    :param graph_names: graph names; default [\'cumulative_return\', \'pred_ic\', \'pred_autocorr\', \'pred_turnover\'].\n    :param show_notebook: whether to display graphics in notebook, the default is `True`.\n    :param show_nature_day: whether to display the abscissa of non-trading day.\n    :param \\*\\*kwargs: contains some parameters to control plot style in plotly. Currently, supports\n       - `rangebreaks`: https://plotly.com/python/time-series/#Hiding-Weekends-and-Holidays\n    :return: if show_notebook is True, display in notebook; else return `plotly.graph_objs.Figure` list.\n    '
    figure_list = []
    for graph_name in graph_names:
        fun_res = eval(f'_{graph_name}')(pred_label=pred_label, lag=lag, N=N, reverse=reverse, rank=rank, show_nature_day=show_nature_day, **kwargs)
        figure_list += fun_res
    if show_notebook:
        BarGraph.show_graph_in_notebook(figure_list)
    else:
        return figure_list