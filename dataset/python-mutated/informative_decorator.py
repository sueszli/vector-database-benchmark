from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union
from pandas import DataFrame
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.strategy.strategy_helper import merge_informative_pair
PopulateIndicators = Callable[[Any, DataFrame, dict], DataFrame]

@dataclass
class InformativeData:
    asset: Optional[str]
    timeframe: str
    fmt: Union[str, Callable[[Any], str], None]
    ffill: bool
    candle_type: Optional[CandleType]

def informative(timeframe: str, asset: str='', fmt: Optional[Union[str, Callable[[Any], str]]]=None, *, candle_type: Optional[Union[CandleType, str]]=None, ffill: bool=True) -> Callable[[PopulateIndicators], PopulateIndicators]:
    if False:
        return 10
    "\n    A decorator for populate_indicators_Nn(self, dataframe, metadata), allowing these functions to\n    define informative indicators.\n\n    Example usage:\n\n        @informative('1h')\n        def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:\n            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)\n            return dataframe\n\n    :param timeframe: Informative timeframe. Must always be equal or higher than strategy timeframe.\n    :param asset: Informative asset, for example BTC, BTC/USDT, ETH/BTC. Do not specify to use\n                  current pair. Also supports limited pair format strings (see below)\n    :param fmt: Column format (str) or column formatter (callable(name, asset, timeframe)). When not\n    specified, defaults to:\n    * {base}_{quote}_{column}_{timeframe} if asset is specified.\n    * {column}_{timeframe} if asset is not specified.\n    Pair format supports these format variables:\n    * {base} - base currency in lower case, for example 'eth'.\n    * {BASE} - same as {base}, except in upper case.\n    * {quote} - quote currency in lower case, for example 'usdt'.\n    * {QUOTE} - same as {quote}, except in upper case.\n    Format string additionally supports this variables.\n    * {asset} - full name of the asset, for example 'BTC/USDT'.\n    * {column} - name of dataframe column.\n    * {timeframe} - timeframe of informative dataframe.\n    :param ffill: ffill dataframe after merging informative pair.\n    :param candle_type: '', mark, index, premiumIndex, or funding_rate\n    "
    _asset = asset
    _timeframe = timeframe
    _fmt = fmt
    _ffill = ffill
    _candle_type = CandleType.from_string(candle_type) if candle_type else None

    def decorator(fn: PopulateIndicators):
        if False:
            return 10
        informative_pairs = getattr(fn, '_ft_informative', [])
        informative_pairs.append(InformativeData(_asset, _timeframe, _fmt, _ffill, _candle_type))
        setattr(fn, '_ft_informative', informative_pairs)
        return fn
    return decorator

def __get_pair_formats(market: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if False:
        while True:
            i = 10
    if not market:
        return {}
    base = market['base']
    quote = market['quote']
    return {'base': base.lower(), 'BASE': base.upper(), 'quote': quote.lower(), 'QUOTE': quote.upper()}

def _format_pair_name(config, pair: str, market: Optional[Dict[str, Any]]=None) -> str:
    if False:
        i = 10
        return i + 15
    return pair.format(stake_currency=config['stake_currency'], stake=config['stake_currency'], **__get_pair_formats(market)).upper()

def _create_and_merge_informative_pair(strategy, dataframe: DataFrame, metadata: dict, inf_data: InformativeData, populate_indicators: PopulateIndicators):
    if False:
        while True:
            i = 10
    asset = inf_data.asset or ''
    timeframe = inf_data.timeframe
    fmt = inf_data.fmt
    candle_type = inf_data.candle_type
    config = strategy.config
    if asset:
        market1 = strategy.dp.market(metadata['pair'])
        asset = _format_pair_name(config, asset, market1)
    else:
        asset = metadata['pair']
    market = strategy.dp.market(asset)
    if market is None:
        raise OperationalException(f'Market {asset} is not available.')
    if not fmt:
        fmt = '{column}_{timeframe}'
        if inf_data.asset:
            fmt = '{base}_{quote}_' + fmt
    inf_metadata = {'pair': asset, 'timeframe': timeframe}
    inf_dataframe = strategy.dp.get_pair_dataframe(asset, timeframe, candle_type)
    inf_dataframe = populate_indicators(strategy, inf_dataframe, inf_metadata)
    formatter: Any = None
    if callable(fmt):
        formatter = fmt
    else:
        formatter = fmt.format
    fmt_args = {**__get_pair_formats(market), 'asset': asset, 'timeframe': timeframe}
    inf_dataframe.rename(columns=lambda column: formatter(column=column, **fmt_args), inplace=True)
    date_column = formatter(column='date', **fmt_args)
    if date_column in dataframe.columns:
        raise OperationalException(f'Duplicate column name {date_column} exists in dataframe! Ensure column names are unique!')
    dataframe = merge_informative_pair(dataframe, inf_dataframe, strategy.timeframe, timeframe, ffill=inf_data.ffill, append_timeframe=False, date_column=date_column)
    return dataframe