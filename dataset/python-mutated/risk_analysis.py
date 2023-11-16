from typing import Iterable
import pandas as pd
import plotly.graph_objs as py
from ...evaluate import risk_analysis
from ..graph import SubplotsGraph, ScatterGraph

def _get_risk_analysis_data_with_report(report_normal_df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    if False:
        while True:
            i = 10
    'Get risk analysis data with report\n\n    :param report_normal_df: report data\n    :param report_long_short_df: report data\n    :param date: date string\n    :return:\n    '
    analysis = dict()
    if not report_normal_df.empty:
        analysis['excess_return_without_cost'] = risk_analysis(report_normal_df['return'] - report_normal_df['bench'])
        analysis['excess_return_with_cost'] = risk_analysis(report_normal_df['return'] - report_normal_df['bench'] - report_normal_df['cost'])
    analysis_df = pd.concat(analysis)
    analysis_df['date'] = date
    return analysis_df

def _get_all_risk_analysis(risk_df: pd.DataFrame) -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    'risk_df to standard\n\n    :param risk_df: risk data\n    :return:\n    '
    if risk_df is None:
        return pd.DataFrame()
    risk_df = risk_df.unstack()
    risk_df.columns = risk_df.columns.droplevel(0)
    return risk_df.drop('mean', axis=1)

def _get_monthly_risk_analysis_with_report(report_normal_df: pd.DataFrame) -> pd.DataFrame:
    if False:
        i = 10
        return i + 15
    'Get monthly analysis data\n\n    :param report_normal_df:\n    # :param report_long_short_df:\n    :return:\n    '
    report_normal_gp = report_normal_df.groupby([report_normal_df.index.year, report_normal_df.index.month])
    gp_month = sorted(set(report_normal_gp.size().index))
    _monthly_df = pd.DataFrame()
    for gp_m in gp_month:
        _m_report_normal = report_normal_gp.get_group(gp_m)
        if len(_m_report_normal) < 3:
            continue
        month_days = pd.Timestamp(year=gp_m[0], month=gp_m[1], day=1).days_in_month
        _temp_df = _get_risk_analysis_data_with_report(_m_report_normal, pd.Timestamp(year=gp_m[0], month=gp_m[1], day=month_days))
        _monthly_df = pd.concat([_monthly_df, _temp_df], sort=False)
    return _monthly_df

def _get_monthly_analysis_with_feature(monthly_df: pd.DataFrame, feature: str='annualized_return') -> pd.DataFrame:
    if False:
        while True:
            i = 10
    '\n\n    :param monthly_df:\n    :param feature:\n    :return:\n    '
    _monthly_df_gp = monthly_df.reset_index().groupby(['level_1'])
    _name_df = _monthly_df_gp.get_group(feature).set_index(['level_0', 'level_1'])
    _temp_df = _name_df.pivot_table(index='date', values=['risk'], columns=_name_df.index)
    _temp_df.columns = map(lambda x: '_'.join(x[-1]), _temp_df.columns)
    _temp_df.index = _temp_df.index.strftime('%Y-%m')
    return _temp_df

def _get_risk_analysis_figure(analysis_df: pd.DataFrame) -> Iterable[py.Figure]:
    if False:
        i = 10
        return i + 15
    'Get analysis graph figure\n\n    :param analysis_df:\n    :return:\n    '
    if analysis_df is None:
        return []
    _figure = SubplotsGraph(_get_all_risk_analysis(analysis_df), kind_map=dict(kind='BarGraph', kwargs={}), subplots_kwargs={'rows': 1, 'cols': 4}).figure
    return (_figure,)

def _get_monthly_risk_analysis_figure(report_normal_df: pd.DataFrame) -> Iterable[py.Figure]:
    if False:
        i = 10
        return i + 15
    'Get analysis monthly graph figure\n\n    :param report_normal_df:\n    :param report_long_short_df:\n    :return:\n    '
    if report_normal_df is None:
        return []
    _monthly_df = _get_monthly_risk_analysis_with_report(report_normal_df=report_normal_df)
    for _feature in ['annualized_return', 'max_drawdown', 'information_ratio', 'std']:
        _temp_df = _get_monthly_analysis_with_feature(_monthly_df, _feature)
        yield ScatterGraph(_temp_df, layout=dict(title=_feature, xaxis=dict(type='category', tickangle=45)), graph_kwargs={'mode': 'lines+markers'}).figure

def risk_analysis_graph(analysis_df: pd.DataFrame=None, report_normal_df: pd.DataFrame=None, report_long_short_df: pd.DataFrame=None, show_notebook: bool=True) -> Iterable[py.Figure]:
    if False:
        for i in range(10):
            print('nop')
    'Generate analysis graph and monthly analysis\n\n        Example:\n\n\n            .. code-block:: python\n\n                import qlib\n                import pandas as pd\n                from qlib.utils.time import Freq\n                from qlib.utils import flatten_dict\n                from qlib.backtest import backtest, executor\n                from qlib.contrib.evaluate import risk_analysis\n                from qlib.contrib.strategy import TopkDropoutStrategy\n\n                # init qlib\n                qlib.init(provider_uri=<qlib data dir>)\n\n                CSI300_BENCH = "SH000300"\n                FREQ = "day"\n                STRATEGY_CONFIG = {\n                    "topk": 50,\n                    "n_drop": 5,\n                    # pred_score, pd.Series\n                    "signal": pred_score,\n                }\n\n                EXECUTOR_CONFIG = {\n                    "time_per_step": "day",\n                    "generate_portfolio_metrics": True,\n                }\n\n                backtest_config = {\n                    "start_time": "2017-01-01",\n                    "end_time": "2020-08-01",\n                    "account": 100000000,\n                    "benchmark": CSI300_BENCH,\n                    "exchange_kwargs": {\n                        "freq": FREQ,\n                        "limit_threshold": 0.095,\n                        "deal_price": "close",\n                        "open_cost": 0.0005,\n                        "close_cost": 0.0015,\n                        "min_cost": 5,\n                    },\n                }\n\n                # strategy object\n                strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)\n                # executor object\n                executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)\n                # backtest\n                portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)\n                analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))\n                # backtest info\n                report_normal_df, positions_normal = portfolio_metric_dict.get(analysis_freq)\n                analysis = dict()\n                analysis["excess_return_without_cost"] = risk_analysis(\n                    report_normal_df["return"] - report_normal_df["bench"], freq=analysis_freq\n                )\n                analysis["excess_return_with_cost"] = risk_analysis(\n                    report_normal_df["return"] - report_normal_df["bench"] - report_normal_df["cost"], freq=analysis_freq\n                )\n\n                analysis_df = pd.concat(analysis)  # type: pd.DataFrame\n                analysis_position.risk_analysis_graph(analysis_df, report_normal_df)\n\n\n\n    :param analysis_df: analysis data, index is **pd.MultiIndex**; columns names is **[risk]**.\n\n\n            .. code-block:: python\n\n                                                                  risk\n                excess_return_without_cost mean               0.000692\n                                           std                0.005374\n                                           annualized_return  0.174495\n                                           information_ratio  2.045576\n                                           max_drawdown      -0.079103\n                excess_return_with_cost    mean               0.000499\n                                           std                0.005372\n                                           annualized_return  0.125625\n                                           information_ratio  1.473152\n                                           max_drawdown      -0.088263\n\n\n    :param report_normal_df: **df.index.name** must be **date**, df.columns must contain **return**, **turnover**, **cost**, **bench**.\n\n\n            .. code-block:: python\n\n                            return      cost        bench       turnover\n                date\n                2017-01-04  0.003421    0.000864    0.011693    0.576325\n                2017-01-05  0.000508    0.000447    0.000721    0.227882\n                2017-01-06  -0.003321   0.000212    -0.004322   0.102765\n                2017-01-09  0.006753    0.000212    0.006874    0.105864\n                2017-01-10  -0.000416   0.000440    -0.003350   0.208396\n\n\n    :param report_long_short_df: **df.index.name** must be **date**, df.columns contain **long**, **short**, **long_short**.\n\n\n            .. code-block:: python\n\n                            long        short       long_short\n                date\n                2017-01-04  -0.001360   0.001394    0.000034\n                2017-01-05  0.002456    0.000058    0.002514\n                2017-01-06  0.000120    0.002739    0.002859\n                2017-01-09  0.001436    0.001838    0.003273\n                2017-01-10  0.000824    -0.001944   -0.001120\n\n\n    :param show_notebook: Whether to display graphics in a notebook, default **True**.\n        If True, show graph in notebook\n        If False, return graph figure\n    :return:\n    '
    _figure_list = list(_get_risk_analysis_figure(analysis_df)) + list(_get_monthly_risk_analysis_figure(report_normal_df))
    if show_notebook:
        ScatterGraph.show_graph_in_notebook(_figure_list)
    else:
        return _figure_list