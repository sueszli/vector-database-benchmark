"""
Dataprovider
Responsible to provide data to the bot
including ticker and orderbook data, live and historical candle (OHLCV) data
Common Interface for bot and strategy to access data.
"""
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pandas import DataFrame, Timedelta, Timestamp, to_timedelta
from freqtrade.configuration import TimeRange
from freqtrade.constants import FULL_DATAFRAME_THRESHOLD, Config, ListPairsWithTimeframes, PairWithTimeframe
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType, RPCMessageType, RunMode
from freqtrade.exceptions import ExchangeError, OperationalException
from freqtrade.exchange import Exchange, timeframe_to_prev_date, timeframe_to_seconds
from freqtrade.exchange.types import OrderBook
from freqtrade.misc import append_candles_to_dataframe
from freqtrade.rpc import RPCManager
from freqtrade.rpc.rpc_types import RPCAnalyzedDFMsg
from freqtrade.util import PeriodicCache
logger = logging.getLogger(__name__)
NO_EXCHANGE_EXCEPTION = 'Exchange is not available to DataProvider.'
MAX_DATAFRAME_CANDLES = 1000

class DataProvider:

    def __init__(self, config: Config, exchange: Optional[Exchange], pairlists=None, rpc: Optional[RPCManager]=None) -> None:
        if False:
            print('Hello World!')
        self._config = config
        self._exchange = exchange
        self._pairlists = pairlists
        self.__rpc = rpc
        self.__cached_pairs: Dict[PairWithTimeframe, Tuple[DataFrame, datetime]] = {}
        self.__slice_index: Optional[int] = None
        self.__slice_date: Optional[datetime] = None
        self.__cached_pairs_backtesting: Dict[PairWithTimeframe, DataFrame] = {}
        self.__producer_pairs_df: Dict[str, Dict[PairWithTimeframe, Tuple[DataFrame, datetime]]] = {}
        self.__producer_pairs: Dict[str, List[str]] = {}
        self._msg_queue: deque = deque()
        self._default_candle_type = self._config.get('candle_type_def', CandleType.SPOT)
        self._default_timeframe = self._config.get('timeframe', '1h')
        self.__msg_cache = PeriodicCache(maxsize=1000, ttl=timeframe_to_seconds(self._default_timeframe))
        self.producers = self._config.get('external_message_consumer', {}).get('producers', [])
        self.external_data_enabled = len(self.producers) > 0

    def _set_dataframe_max_index(self, limit_index: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        Limit analyzed dataframe to max specified index.\n        Only relevant in backtesting.\n        :param limit_index: dataframe index.\n        '
        self.__slice_index = limit_index

    def _set_dataframe_max_date(self, limit_date: datetime):
        if False:
            print('Hello World!')
        '\n        Limit infomrative dataframe to max specified index.\n        Only relevant in backtesting.\n        :param limit_date: "current date"\n        '
        self.__slice_date = limit_date

    def _set_cached_df(self, pair: str, timeframe: str, dataframe: DataFrame, candle_type: CandleType) -> None:
        if False:
            while True:
                i = 10
        '\n        Store cached Dataframe.\n        Using private method as this should never be used by a user\n        (but the class is exposed via `self.dp` to the strategy)\n        :param pair: pair to get the data for\n        :param timeframe: Timeframe to get data for\n        :param dataframe: analyzed dataframe\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        '
        pair_key = (pair, timeframe, candle_type)
        self.__cached_pairs[pair_key] = (dataframe, datetime.now(timezone.utc))

    def _set_producer_pairs(self, pairlist: List[str], producer_name: str='default'):
        if False:
            print('Hello World!')
        '\n        Set the pairs received to later be used.\n\n        :param pairlist: List of pairs\n        '
        self.__producer_pairs[producer_name] = pairlist

    def get_producer_pairs(self, producer_name: str='default') -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the pairs cached from the producer\n\n        :returns: List of pairs\n        '
        return self.__producer_pairs.get(producer_name, []).copy()

    def _emit_df(self, pair_key: PairWithTimeframe, dataframe: DataFrame, new_candle: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Send this dataframe as an ANALYZED_DF message to RPC\n\n        :param pair_key: PairWithTimeframe tuple\n        :param dataframe: Dataframe to emit\n        :param new_candle: This is a new candle\n        '
        if self.__rpc:
            msg: RPCAnalyzedDFMsg = {'type': RPCMessageType.ANALYZED_DF, 'data': {'key': pair_key, 'df': dataframe.tail(1), 'la': datetime.now(timezone.utc)}}
            self.__rpc.send_msg(msg)
            if new_candle:
                self.__rpc.send_msg({'type': RPCMessageType.NEW_CANDLE, 'data': pair_key})

    def _replace_external_df(self, pair: str, dataframe: DataFrame, last_analyzed: datetime, timeframe: str, candle_type: CandleType, producer_name: str='default') -> None:
        if False:
            while True:
                i = 10
        '\n        Add the pair data to this class from an external source.\n\n        :param pair: pair to get the data for\n        :param timeframe: Timeframe to get data for\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        '
        pair_key = (pair, timeframe, candle_type)
        if producer_name not in self.__producer_pairs_df:
            self.__producer_pairs_df[producer_name] = {}
        _last_analyzed = datetime.now(timezone.utc) if not last_analyzed else last_analyzed
        self.__producer_pairs_df[producer_name][pair_key] = (dataframe, _last_analyzed)
        logger.debug(f'External DataFrame for {pair_key} from {producer_name} added.')

    def _add_external_df(self, pair: str, dataframe: DataFrame, last_analyzed: datetime, timeframe: str, candle_type: CandleType, producer_name: str='default') -> Tuple[bool, int]:
        if False:
            return 10
        '\n        Append a candle to the existing external dataframe. The incoming dataframe\n        must have at least 1 candle.\n\n        :param pair: pair to get the data for\n        :param timeframe: Timeframe to get data for\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        :returns: False if the candle could not be appended, or the int number of missing candles.\n        '
        pair_key = (pair, timeframe, candle_type)
        if dataframe.empty:
            return (False, 0)
        if len(dataframe) >= FULL_DATAFRAME_THRESHOLD:
            self._replace_external_df(pair, dataframe, last_analyzed=last_analyzed, timeframe=timeframe, candle_type=candle_type, producer_name=producer_name)
            return (True, 0)
        if producer_name not in self.__producer_pairs_df or pair_key not in self.__producer_pairs_df[producer_name]:
            return (False, 1000)
        (existing_df, _) = self.__producer_pairs_df[producer_name][pair_key]
        timeframe_delta: Timedelta = to_timedelta(timeframe)
        local_last: Timestamp = existing_df.iloc[-1]['date']
        incoming_first: Timestamp = dataframe.iloc[0]['date']
        existing_df1 = existing_df[existing_df['date'] < incoming_first]
        candle_difference = (incoming_first - local_last) / timeframe_delta
        if candle_difference > 1:
            return (False, int(candle_difference))
        if existing_df1.empty:
            appended_df = dataframe
        else:
            appended_df = append_candles_to_dataframe(existing_df1, dataframe)
        self._replace_external_df(pair, appended_df, last_analyzed=last_analyzed, timeframe=timeframe, candle_type=candle_type, producer_name=producer_name)
        return (True, 0)

    def get_producer_df(self, pair: str, timeframe: Optional[str]=None, candle_type: Optional[CandleType]=None, producer_name: str='default') -> Tuple[DataFrame, datetime]:
        if False:
            return 10
        '\n        Get the pair data from producers.\n\n        :param pair: pair to get the data for\n        :param timeframe: Timeframe to get data for\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        :returns: Tuple of the DataFrame and last analyzed timestamp\n        '
        _timeframe = self._default_timeframe if not timeframe else timeframe
        _candle_type = self._default_candle_type if not candle_type else candle_type
        pair_key = (pair, _timeframe, _candle_type)
        if producer_name not in self.__producer_pairs_df:
            return (DataFrame(), datetime.fromtimestamp(0, tz=timezone.utc))
        if pair_key not in self.__producer_pairs_df[producer_name]:
            return (DataFrame(), datetime.fromtimestamp(0, tz=timezone.utc))
        (df, la) = self.__producer_pairs_df[producer_name][pair_key]
        return (df.copy(), la)

    def add_pairlisthandler(self, pairlists) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Allow adding pairlisthandler after initialization\n        '
        self._pairlists = pairlists

    def historic_ohlcv(self, pair: str, timeframe: str, candle_type: str='') -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        "\n        Get stored historical candle (OHLCV) data\n        :param pair: pair to get the data for\n        :param timeframe: timeframe to get data for\n        :param candle_type: '', mark, index, premiumIndex, or funding_rate\n        "
        _candle_type = CandleType.from_string(candle_type) if candle_type != '' else self._config['candle_type_def']
        saved_pair: PairWithTimeframe = (pair, str(timeframe), _candle_type)
        if saved_pair not in self.__cached_pairs_backtesting:
            timerange = TimeRange.parse_timerange(None if self._config.get('timerange') is None else str(self._config.get('timerange')))
            startup_candles = self.get_required_startup(str(timeframe), False)
            tf_seconds = timeframe_to_seconds(str(timeframe))
            timerange.subtract_start(tf_seconds * startup_candles)
            self.__cached_pairs_backtesting[saved_pair] = load_pair_history(pair=pair, timeframe=timeframe, datadir=self._config['datadir'], timerange=timerange, data_format=self._config['dataformat_ohlcv'], candle_type=_candle_type)
        return self.__cached_pairs_backtesting[saved_pair].copy()

    def get_required_startup(self, timeframe: str, add_train_candles: bool=True) -> int:
        if False:
            return 10
        freqai_config = self._config.get('freqai', {})
        if not freqai_config.get('enabled', False):
            return self._config.get('startup_candle_count', 0)
        else:
            startup_candles = self._config.get('startup_candle_count', 0)
            indicator_periods = freqai_config['feature_parameters']['indicator_periods_candles']
            self._config['startup_candle_count'] = max(startup_candles, max(indicator_periods))
            tf_seconds = timeframe_to_seconds(timeframe)
            train_candles = 0
            if add_train_candles:
                train_candles = freqai_config['train_period_days'] * 86400 / tf_seconds
            total_candles = int(self._config['startup_candle_count'] + train_candles)
            logger.info(f'Increasing startup_candle_count for freqai to {total_candles}')
            return total_candles

    def get_pair_dataframe(self, pair: str, timeframe: Optional[str]=None, candle_type: str='') -> DataFrame:
        if False:
            while True:
                i = 10
        "\n        Return pair candle (OHLCV) data, either live or cached historical -- depending\n        on the runmode.\n        Only combinations in the pairlist or which have been specified as informative pairs\n        will be available.\n        :param pair: pair to get the data for\n        :param timeframe: timeframe to get data for\n        :return: Dataframe for this pair\n        :param candle_type: '', mark, index, premiumIndex, or funding_rate\n        "
        if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
            data = self.ohlcv(pair=pair, timeframe=timeframe, candle_type=candle_type)
        else:
            timeframe = timeframe or self._config['timeframe']
            data = self.historic_ohlcv(pair=pair, timeframe=timeframe, candle_type=candle_type)
            if self.__slice_date:
                cutoff_date = timeframe_to_prev_date(timeframe, self.__slice_date)
                data = data.loc[data['date'] < cutoff_date]
        if len(data) == 0:
            logger.warning(f'No data found for ({pair}, {timeframe}, {candle_type}).')
        return data

    def get_analyzed_dataframe(self, pair: str, timeframe: str) -> Tuple[DataFrame, datetime]:
        if False:
            i = 10
            return i + 15
        '\n        Retrieve the analyzed dataframe. Returns the full dataframe in trade mode (live / dry),\n        and the last 1000 candles (up to the time evaluated at this moment) in all other modes.\n        :param pair: pair to get the data for\n        :param timeframe: timeframe to get data for\n        :return: Tuple of (Analyzed Dataframe, lastrefreshed) for the requested pair / timeframe\n            combination.\n            Returns empty dataframe and Epoch 0 (1970-01-01) if no dataframe was cached.\n        '
        pair_key = (pair, timeframe, self._config.get('candle_type_def', CandleType.SPOT))
        if pair_key in self.__cached_pairs:
            if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
                (df, date) = self.__cached_pairs[pair_key]
            else:
                (df, date) = self.__cached_pairs[pair_key]
                if self.__slice_index is not None:
                    max_index = self.__slice_index
                    df = df.iloc[max(0, max_index - MAX_DATAFRAME_CANDLES):max_index]
            return (df, date)
        else:
            return (DataFrame(), datetime.fromtimestamp(0, tz=timezone.utc))

    @property
    def runmode(self) -> RunMode:
        if False:
            while True:
                i = 10
        '\n        Get runmode of the bot\n        can be "live", "dry-run", "backtest", "edgecli", "hyperopt" or "other".\n        '
        return RunMode(self._config.get('runmode', RunMode.OTHER))

    def current_whitelist(self) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        fetch latest available whitelist.\n\n        Useful when you have a large whitelist and need to call each pair as an informative pair.\n        As available pairs does not show whitelist until after informative pairs have been cached.\n        :return: list of pairs in whitelist\n        '
        if self._pairlists:
            return self._pairlists.whitelist.copy()
        else:
            raise OperationalException('Dataprovider was not initialized with a pairlist provider.')

    def clear_cache(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clear pair dataframe cache.\n        '
        self.__cached_pairs = {}
        self.__slice_index = 0

    def refresh(self, pairlist: ListPairsWithTimeframes, helping_pairs: Optional[ListPairsWithTimeframes]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Refresh data, called with each cycle\n        '
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        final_pairs = pairlist + helping_pairs if helping_pairs else pairlist
        self._exchange.refresh_latest_ohlcv(final_pairs)

    @property
    def available_pairs(self) -> ListPairsWithTimeframes:
        if False:
            return 10
        '\n        Return a list of tuples containing (pair, timeframe) for which data is currently cached.\n        Should be whitelist + open trades.\n        '
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        return list(self._exchange._klines.keys())

    def ohlcv(self, pair: str, timeframe: Optional[str]=None, copy: bool=True, candle_type: str='') -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        "\n        Get candle (OHLCV) data for the given pair as DataFrame\n        Please use the `available_pairs` method to verify which pairs are currently cached.\n        :param pair: pair to get the data for\n        :param timeframe: Timeframe to get data for\n        :param candle_type: '', mark, index, premiumIndex, or funding_rate\n        :param copy: copy dataframe before returning if True.\n                     Use False only for read-only operations (where the dataframe is not modified)\n        "
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
            _candle_type = CandleType.from_string(candle_type) if candle_type != '' else self._config['candle_type_def']
            return self._exchange.klines((pair, timeframe or self._config['timeframe'], _candle_type), copy=copy)
        else:
            return DataFrame()

    def market(self, pair: str) -> Optional[Dict[str, Any]]:
        if False:
            return 10
        '\n        Return market data for the pair\n        :param pair: Pair to get the data for\n        :return: Market data dict from ccxt or None if market info is not available for the pair\n        '
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        return self._exchange.markets.get(pair)

    def ticker(self, pair: str):
        if False:
            while True:
                i = 10
        '\n        Return last ticker data from exchange\n        :param pair: Pair to get the data for\n        :return: Ticker dict from exchange or empty dict if ticker is not available for the pair\n        '
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        try:
            return self._exchange.fetch_ticker(pair)
        except ExchangeError:
            return {}

    def orderbook(self, pair: str, maximum: int) -> OrderBook:
        if False:
            i = 10
            return i + 15
        '\n        Fetch latest l2 orderbook data\n        Warning: Does a network request - so use with common sense.\n        :param pair: pair to get the data for\n        :param maximum: Maximum number of orderbook entries to query\n        :return: dict including bids/asks with a total of `maximum` entries.\n        '
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        return self._exchange.fetch_l2_order_book(pair, maximum)

    def send_msg(self, message: str, *, always_send: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Send custom RPC Notifications from your bot.\n        Will not send any bot in modes other than Dry-run or Live.\n        :param message: Message to be sent. Must be below 4096.\n        :param always_send: If False, will send the message only once per candle, and surpress\n                            identical messages.\n                            Careful as this can end up spaming your chat.\n                            Defaults to False\n        '
        if self.runmode not in (RunMode.DRY_RUN, RunMode.LIVE):
            return
        if always_send or message not in self.__msg_cache:
            self._msg_queue.append(message)
        self.__msg_cache[message] = True