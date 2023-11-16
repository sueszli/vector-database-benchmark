from typing import Optional
import pandas as pd
from freqtrade.exchange import timeframe_to_minutes

def merge_informative_pair(dataframe: pd.DataFrame, informative: pd.DataFrame, timeframe: str, timeframe_inf: str, ffill: bool=True, append_timeframe: bool=True, date_column: str='date', suffix: Optional[str]=None) -> pd.DataFrame:
    if False:
        print('Hello World!')
    "\n    Correctly merge informative samples to the original dataframe, avoiding lookahead bias.\n\n    Since dates are candle open dates, merging a 15m candle that starts at 15:00, and a\n    1h candle that starts at 15:00 will result in all candles to know the close at 16:00\n    which they should not know.\n\n    Moves the date of the informative pair by 1 time interval forward.\n    This way, the 14:00 1h candle is merged to 15:00 15m candle, since the 14:00 1h candle is the\n    last candle that's closed at 15:00, 15:15, 15:30 or 15:45.\n\n    Assuming inf_tf = '1d' - then the resulting columns will be:\n    date_1d, open_1d, high_1d, low_1d, close_1d, rsi_1d\n\n    :param dataframe: Original dataframe\n    :param informative: Informative pair, most likely loaded via dp.get_pair_dataframe\n    :param timeframe: Timeframe of the original pair sample.\n    :param timeframe_inf: Timeframe of the informative pair sample.\n    :param ffill: Forwardfill missing values - optional but usually required\n    :param append_timeframe: Rename columns by appending timeframe.\n    :param date_column: A custom date column name.\n    :param suffix: A string suffix to add at the end of the informative columns. If specified,\n                   append_timeframe must be false.\n    :return: Merged dataframe\n    :raise: ValueError if the secondary timeframe is shorter than the dataframe timeframe\n    "
    minutes_inf = timeframe_to_minutes(timeframe_inf)
    minutes = timeframe_to_minutes(timeframe)
    if minutes == minutes_inf:
        informative['date_merge'] = informative[date_column]
    elif minutes < minutes_inf:
        if not informative.empty:
            informative['date_merge'] = informative[date_column] + pd.to_timedelta(minutes_inf, 'm') - pd.to_timedelta(minutes, 'm')
        else:
            informative['date_merge'] = informative[date_column]
    else:
        raise ValueError('Tried to merge a faster timeframe to a slower timeframe.This would create new rows, and can throw off your regular indicators.')
    date_merge = 'date_merge'
    if suffix and append_timeframe:
        raise ValueError('You can not specify `append_timeframe` as True and a `suffix`.')
    elif append_timeframe:
        date_merge = f'date_merge_{timeframe_inf}'
        informative.columns = [f'{col}_{timeframe_inf}' for col in informative.columns]
    elif suffix:
        date_merge = f'date_merge_{suffix}'
        informative.columns = [f'{col}_{suffix}' for col in informative.columns]
    if ffill:
        dataframe = pd.merge_ordered(dataframe, informative, fill_method='ffill', left_on='date', right_on=date_merge, how='left')
    else:
        dataframe = pd.merge(dataframe, informative, left_on='date', right_on=date_merge, how='left')
    dataframe = dataframe.drop(date_merge, axis=1)
    return dataframe

def stoploss_from_open(open_relative_stop: float, current_profit: float, is_short: bool=False, leverage: float=1.0) -> float:
    if False:
        return 10
    '\n    Given the current profit, and a desired stop loss value relative to the trade entry price,\n    return a stop loss value that is relative to the current price, and which can be\n    returned from `custom_stoploss`.\n\n    The requested stop can be positive for a stop above the open price, or negative for\n    a stop below the open price. The return value is always >= 0.\n    `open_relative_stop` will be considered as adjusted for leverage if leverage is provided..\n\n    Returns 0 if the resulting stop price would be above/below (longs/shorts) the current price\n\n    :param open_relative_stop: Desired stop loss percentage, relative to the open price,\n                               adjusted for leverage\n    :param current_profit: The current profit percentage\n    :param is_short: When true, perform the calculation for short instead of long\n    :param leverage: Leverage to use for the calculation\n    :return: Stop loss value relative to current price\n    '
    _current_profit = current_profit / leverage
    if _current_profit == -1 and (not is_short) or (is_short and _current_profit == 1):
        return 1
    if is_short is True:
        stoploss = -1 + (1 - open_relative_stop / leverage) / (1 - _current_profit)
    else:
        stoploss = 1 - (1 + open_relative_stop / leverage) / (1 + _current_profit)
    return max(stoploss * leverage, 0.0)

def stoploss_from_absolute(stop_rate: float, current_rate: float, is_short: bool=False, leverage: float=1.0) -> float:
    if False:
        print('Hello World!')
    '\n    Given current price and desired stop price, return a stop loss value that is relative to current\n    price.\n\n    The requested stop can be positive for a stop above the open price, or negative for\n    a stop below the open price. The return value is always >= 0.\n\n    Returns 0 if the resulting stop price would be above the current price.\n\n    :param stop_rate: Stop loss price.\n    :param current_rate: Current asset price.\n    :param is_short: When true, perform the calculation for short instead of long\n    :param leverage: Leverage to use for the calculation\n    :return: Positive stop loss value relative to current price\n    '
    if current_rate == 0:
        return 1
    stoploss = 1 - stop_rate / current_rate
    if is_short:
        stoploss = -stoploss
    return max(min(stoploss, 1.0), 0.0) * leverage