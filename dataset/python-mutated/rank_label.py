import copy
from typing import Iterable
import pandas as pd
import plotly.graph_objs as go
from ..graph import ScatterGraph
from ..analysis_position.parse_position import get_position_data

def _get_figure_with_position(position: dict, label_data: pd.DataFrame, start_date=None, end_date=None) -> Iterable[go.Figure]:
    if False:
        for i in range(10):
            print('nop')
    'Get average analysis figures\n\n    :param position: position\n    :param label_data:\n    :param start_date:\n    :param end_date:\n    :return:\n    '
    _position_df = get_position_data(position, label_data, calculate_label_rank=True, start_date=start_date, end_date=end_date)
    res_dict = dict()
    _pos_gp = _position_df.groupby(level=1)
    for _item in _pos_gp:
        _date = _item[0]
        _day_df = _item[1]
        _day_value = res_dict.setdefault(_date, {})
        for (_i, _name) in {0: 'Hold', 1: 'Buy', -1: 'Sell'}.items():
            _temp_df = _day_df[_day_df['status'] == _i]
            if _temp_df.empty:
                _day_value[_name] = 0
            else:
                _day_value[_name] = _temp_df['rank_label_mean'].values[0]
    _res_df = pd.DataFrame.from_dict(res_dict, orient='index')
    _res_df.index = _res_df.index.strftime('%Y-%m-%d')
    for _col in _res_df.columns:
        yield ScatterGraph(_res_df.loc[:, [_col]], layout=dict(title=_col, xaxis=dict(type='category', tickangle=45), yaxis=dict(title='lable-rank-ratio: %')), graph_kwargs=dict(mode='lines+markers')).figure

def rank_label_graph(position: dict, label_data: pd.DataFrame, start_date=None, end_date=None, show_notebook=True) -> Iterable[go.Figure]:
    if False:
        i = 10
        return i + 15
    "Ranking percentage of stocks buy, sell, and holding on the trading day.\n    Average rank-ratio(similar to **sell_df['label'].rank(ascending=False) / len(sell_df)**) of daily trading\n\n        Example:\n\n\n            .. code-block:: python\n\n                from qlib.data import D\n                from qlib.contrib.evaluate import backtest\n                from qlib.contrib.strategy import TopkDropoutStrategy\n\n                # backtest parameters\n                bparas = {}\n                bparas['limit_threshold'] = 0.095\n                bparas['account'] = 1000000000\n\n                sparas = {}\n                sparas['topk'] = 50\n                sparas['n_drop'] = 230\n                strategy = TopkDropoutStrategy(**sparas)\n\n                _, positions = backtest(pred_df, strategy, **bparas)\n\n                pred_df_dates = pred_df.index.get_level_values(level='datetime')\n                features_df = D.features(D.instruments('csi500'), ['Ref($close, -1)/$close-1'], pred_df_dates.min(), pred_df_dates.max())\n                features_df.columns = ['label']\n\n                qcr.analysis_position.rank_label_graph(positions, features_df, pred_df_dates.min(), pred_df_dates.max())\n\n\n    :param position: position data; **qlib.backtest.backtest** result.\n    :param label_data: **D.features** result; index is **pd.MultiIndex**, index name is **[instrument, datetime]**; columns names is **[label]**.\n\n        **The label T is the change from T to T+1**, it is recommended to use ``close``, example: `D.features(D.instruments('csi500'), ['Ref($close, -1)/$close-1'])`.\n\n\n            .. code-block:: python\n\n                                                label\n                instrument  datetime\n                SH600004        2017-12-11  -0.013502\n                                2017-12-12  -0.072367\n                                2017-12-13  -0.068605\n                                2017-12-14  0.012440\n                                2017-12-15  -0.102778\n\n\n    :param start_date: start date\n    :param end_date: end_date\n    :param show_notebook: **True** or **False**. If True, show graph in notebook, else return figures.\n    :return:\n    "
    position = copy.deepcopy(position)
    label_data.columns = ['label']
    _figures = _get_figure_with_position(position, label_data, start_date, end_date)
    if show_notebook:
        ScatterGraph.show_graph_in_notebook(_figures)
    else:
        return _figures