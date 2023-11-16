import pandas as pd
from ....backtest.profit_attribution import get_stock_weight_df

def parse_position(position: dict=None) -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    'Parse position dict to position DataFrame\n\n    :param position: position data\n    :return: position DataFrame;\n\n\n        .. code-block:: python\n\n            position_df = parse_position(positions)\n            print(position_df.head())\n            # status: 0-hold, -1-sell, 1-buy\n\n                                        amount      cash      count    price status weight\n            instrument  datetime\n            SZ000547    2017-01-04  44.154290   211405.285654   1   205.189575  1   0.031255\n            SZ300202    2017-01-04  60.638845   211405.285654   1   154.356506  1   0.032290\n            SH600158    2017-01-04  46.531681   211405.285654   1   153.895142  1   0.024704\n            SH600545    2017-01-04  197.173093  211405.285654   1   48.607037   1   0.033063\n            SZ000930    2017-01-04  103.938300  211405.285654   1   80.759453   1   0.028958\n\n\n    '
    position_weight_df = get_stock_weight_df(position)
    position_weight_df.fillna(method='ffill', inplace=True)
    previous_data = {'date': None, 'code_list': []}
    result_df = pd.DataFrame()
    for (_trading_date, _value) in position.items():
        _value = _value.position
        _cash = _value.pop('cash')
        for _item in ['now_account_value']:
            if _item in _value:
                _value.pop(_item)
        _trading_day_df = pd.DataFrame.from_dict(_value, orient='index')
        _trading_day_df['weight'] = position_weight_df.loc[_trading_date]
        _trading_day_df['cash'] = _cash
        _trading_day_df['date'] = _trading_date
        _trading_day_df['status'] = 0
        _cur_day_sell = set(previous_data['code_list']) - set(_trading_day_df.index)
        _cur_day_buy = set(_trading_day_df.index) - set(previous_data['code_list'])
        _trading_day_df.loc[_trading_day_df.index.isin(_cur_day_buy), 'status'] = 1
        if not result_df.empty:
            _trading_day_sell_df = result_df.loc[(result_df['date'] == previous_data['date']) & result_df.index.isin(_cur_day_sell)].copy()
            if not _trading_day_sell_df.empty:
                _trading_day_sell_df['status'] = -1
                _trading_day_sell_df['date'] = _trading_date
                _trading_day_df = pd.concat([_trading_day_df, _trading_day_sell_df], sort=False)
        result_df = pd.concat([result_df, _trading_day_df], sort=True)
        previous_data = dict(date=_trading_date, code_list=_trading_day_df[_trading_day_df['status'] != -1].index)
    result_df.reset_index(inplace=True)
    result_df.rename(columns={'date': 'datetime', 'index': 'instrument'}, inplace=True)
    return result_df.set_index(['instrument', 'datetime'])

def _add_label_to_position(position_df: pd.DataFrame, label_data: pd.DataFrame) -> pd.DataFrame:
    if False:
        print('Hello World!')
    'Concat position with custom label\n\n    :param position_df: position DataFrame\n    :param label_data:\n    :return: concat result\n    '
    _start_time = position_df.index.get_level_values(level='datetime').min()
    _end_time = position_df.index.get_level_values(level='datetime').max()
    label_data = label_data.loc(axis=0)[:, pd.to_datetime(_start_time):]
    _result_df = pd.concat([position_df, label_data], axis=1, sort=True).reindex(label_data.index)
    _result_df = _result_df.loc[_result_df.index.get_level_values(1) <= _end_time]
    return _result_df

def _add_bench_to_position(position_df: pd.DataFrame=None, bench: pd.Series=None) -> pd.DataFrame:
    if False:
        return 10
    'Concat position with bench\n\n    :param position_df: position DataFrame\n    :param bench: report normal data\n    :return: concat result\n    '
    _temp_df = position_df.reset_index(level='instrument')
    _temp_df['bench'] = bench.shift(-1)
    res_df = _temp_df.set_index(['instrument', _temp_df.index])
    return res_df

def _calculate_label_rank(df: pd.DataFrame) -> pd.DataFrame:
    if False:
        print('Hello World!')
    'calculate label rank\n\n    :param df:\n    :return:\n    '
    _label_name = 'label'

    def _calculate_day_value(g_df: pd.DataFrame):
        if False:
            return 10
        g_df = g_df.copy()
        g_df['rank_ratio'] = g_df[_label_name].rank(ascending=False) / len(g_df) * 100
        for i in [-1, 0, 1]:
            g_df.loc[g_df['status'] == i, 'rank_label_mean'] = g_df[g_df['status'] == i]['rank_ratio'].mean()
        g_df['excess_return'] = g_df[_label_name] - g_df[_label_name].mean()
        return g_df
    return df.groupby(level='datetime').apply(_calculate_day_value)

def get_position_data(position: dict, label_data: pd.DataFrame, report_normal: pd.DataFrame=None, calculate_label_rank=False, start_date=None, end_date=None) -> pd.DataFrame:
    if False:
        while True:
            i = 10
    "Concat position data with pred/report_normal\n\n    :param position: position data\n    :param report_normal: report normal, must be container 'bench' column\n    :param label_data:\n    :param calculate_label_rank:\n    :param start_date: start date\n    :param end_date: end date\n    :return: concat result,\n        columns: ['amount', 'cash', 'count', 'price', 'status', 'weight', 'label',\n                    'rank_ratio', 'rank_label_mean', 'excess_return', 'score', 'bench']\n        index: ['instrument', 'date']\n    "
    _position_df = parse_position(position)
    _position_df = _add_label_to_position(_position_df, label_data)
    if calculate_label_rank:
        _position_df = _calculate_label_rank(_position_df)
    if report_normal is not None:
        _position_df = _add_bench_to_position(_position_df, report_normal['bench'])
    _date_list = _position_df.index.get_level_values(level='datetime')
    start_date = _date_list.min() if start_date is None else start_date
    end_date = _date_list.max() if end_date is None else end_date
    _position_df = _position_df.loc[(start_date <= _date_list) & (_date_list <= end_date)]
    return _position_df