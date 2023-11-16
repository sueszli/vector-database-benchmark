import pandas as pd
from ..graph import ScatterGraph
from ..utils import guess_plotly_rangebreaks

def _get_score_ic(pred_label: pd.DataFrame):
    if False:
        print('Hello World!')
    '\n\n    :param pred_label:\n    :return:\n    '
    concat_data = pred_label.copy()
    concat_data.dropna(axis=0, how='any', inplace=True)
    _ic = concat_data.groupby(level='datetime').apply(lambda x: x['label'].corr(x['score']))
    _rank_ic = concat_data.groupby(level='datetime').apply(lambda x: x['label'].corr(x['score'], method='spearman'))
    return pd.DataFrame({'ic': _ic, 'rank_ic': _rank_ic})

def score_ic_graph(pred_label: pd.DataFrame, show_notebook: bool=True, **kwargs) -> [list, tuple]:
    if False:
        for i in range(10):
            print('nop')
    "score IC\n\n        Example:\n\n\n            .. code-block:: python\n\n                from qlib.data import D\n                from qlib.contrib.report import analysis_position\n                pred_df_dates = pred_df.index.get_level_values(level='datetime')\n                features_df = D.features(D.instruments('csi500'), ['Ref($close, -2)/Ref($close, -1)-1'], pred_df_dates.min(), pred_df_dates.max())\n                features_df.columns = ['label']\n                pred_label = pd.concat([features_df, pred], axis=1, sort=True).reindex(features_df.index)\n                analysis_position.score_ic_graph(pred_label)\n\n\n    :param pred_label: index is **pd.MultiIndex**, index name is **[instrument, datetime]**; columns names is **[score, label]**.\n\n\n            .. code-block:: python\n\n                instrument  datetime        score         label\n                SH600004  2017-12-11     -0.013502       -0.013502\n                            2017-12-12   -0.072367       -0.072367\n                            2017-12-13   -0.068605       -0.068605\n                            2017-12-14    0.012440        0.012440\n                            2017-12-15   -0.102778       -0.102778\n\n\n    :param show_notebook: whether to display graphics in notebook, the default is **True**.\n    :return: if show_notebook is True, display in notebook; else return **plotly.graph_objs.Figure** list.\n    "
    _ic_df = _get_score_ic(pred_label)
    _figure = ScatterGraph(_ic_df, layout=dict(title='Score IC', xaxis=dict(tickangle=45, rangebreaks=kwargs.get('rangebreaks', guess_plotly_rangebreaks(_ic_df.index)))), graph_kwargs={'mode': 'lines+markers'}).figure
    if show_notebook:
        ScatterGraph.show_graph_in_notebook([_figure])
    else:
        return (_figure,)