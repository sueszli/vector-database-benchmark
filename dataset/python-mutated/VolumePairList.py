"""
Volume PairList provider

Provides dynamic pair list based on trade volumes
"""
import logging
from datetime import timedelta
from typing import Any, Dict, List, Literal
from cachetools import TTLCache
from freqtrade.constants import Config, ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.exchange.types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter
from freqtrade.util import dt_now, format_ms_time
logger = logging.getLogger(__name__)
SORT_VALUES = ['quoteVolume']

class VolumePairList(IPairList):
    is_pairlist_generator = True

    def __init__(self, exchange, pairlistmanager, config: Config, pairlistconfig: Dict[str, Any], pairlist_pos: int) -> None:
        if False:
            return 10
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)
        if 'number_assets' not in self._pairlistconfig:
            raise OperationalException('`number_assets` not specified. Please check your configuration for "pairlist.config.number_assets"')
        self._stake_currency = config['stake_currency']
        self._number_pairs = self._pairlistconfig['number_assets']
        self._sort_key: Literal['quoteVolume'] = self._pairlistconfig.get('sort_key', 'quoteVolume')
        self._min_value = self._pairlistconfig.get('min_value', 0)
        self._refresh_period = self._pairlistconfig.get('refresh_period', 1800)
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._lookback_days = self._pairlistconfig.get('lookback_days', 0)
        self._lookback_timeframe = self._pairlistconfig.get('lookback_timeframe', '1d')
        self._lookback_period = self._pairlistconfig.get('lookback_period', 0)
        self._def_candletype = self._config['candle_type_def']
        if (self._lookback_days > 0) & (self._lookback_period > 0):
            raise OperationalException('Ambigous configuration: lookback_days and lookback_period both set in pairlist config. Please set lookback_days only or lookback_period and lookback_timeframe and restart the bot.')
        if self._lookback_days > 0:
            self._lookback_timeframe = '1d'
            self._lookback_period = self._lookback_days
        self._tf_in_min = timeframe_to_minutes(self._lookback_timeframe)
        self._tf_in_sec = self._tf_in_min * 60
        self._use_range = (self._tf_in_min > 0) & (self._lookback_period > 0)
        if self._use_range & (self._refresh_period < self._tf_in_sec):
            raise OperationalException(f'Refresh period of {self._refresh_period} seconds is smaller than one timeframe of {self._lookback_timeframe}. Please adjust refresh_period to at least {self._tf_in_sec} and restart the bot.')
        if not self._use_range and (not (self._exchange.exchange_has('fetchTickers') and self._exchange.get_option('tickers_have_quoteVolume'))):
            raise OperationalException('Exchange does not support dynamic whitelist in this configuration. Please edit your config and either remove Volumepairlist, or switch to using candles. and restart the bot.')
        if not self._validate_keys(self._sort_key):
            raise OperationalException(f'key {self._sort_key} not in {SORT_VALUES}')
        candle_limit = exchange.ohlcv_candle_limit(self._lookback_timeframe, self._config['candle_type_def'])
        if self._lookback_period < 0:
            raise OperationalException('VolumeFilter requires lookback_period to be >= 0')
        if self._lookback_period > candle_limit:
            raise OperationalException(f'VolumeFilter requires lookback_period to not exceed exchange max request size ({candle_limit})')

    @property
    def needstickers(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Boolean property defining if tickers are necessary.\n        If no Pairlist requires tickers, an empty Dict is passed\n        as tickers argument to filter_pairlist\n        '
        return not self._use_range

    def _validate_keys(self, key):
        if False:
            print('Hello World!')
        return key in SORT_VALUES

    def short_desc(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Short whitelist method description - used for startup-messages\n        '
        return f"{self.name} - top {self._pairlistconfig['number_assets']} volume pairs."

    @staticmethod
    def description() -> str:
        if False:
            while True:
                i = 10
        return 'Provides dynamic pair list based on trade volumes.'

    @staticmethod
    def available_parameters() -> Dict[str, PairlistParameter]:
        if False:
            for i in range(10):
                print('nop')
        return {'number_assets': {'type': 'number', 'default': 30, 'description': 'Number of assets', 'help': 'Number of assets to use from the pairlist'}, 'sort_key': {'type': 'option', 'default': 'quoteVolume', 'options': SORT_VALUES, 'description': 'Sort key', 'help': 'Sort key to use for sorting the pairlist.'}, 'min_value': {'type': 'number', 'default': 0, 'description': 'Minimum value', 'help': 'Minimum value to use for filtering the pairlist.'}, **IPairList.refresh_period_parameter(), 'lookback_days': {'type': 'number', 'default': 0, 'description': 'Lookback Days', 'help': 'Number of days to look back at.'}, 'lookback_timeframe': {'type': 'string', 'default': '', 'description': 'Lookback Timeframe', 'help': 'Timeframe to use for lookback.'}, 'lookback_period': {'type': 'number', 'default': 0, 'description': 'Lookback Period', 'help': 'Number of periods to look back at.'}}

    def gen_pairlist(self, tickers: Tickers) -> List[str]:
        if False:
            i = 10
            return i + 15
        '\n        Generate the pairlist\n        :param tickers: Tickers (from exchange.get_tickers). May be cached.\n        :return: List of pairs\n        '
        pairlist = self._pair_cache.get('pairlist')
        if pairlist:
            return pairlist.copy()
        else:
            _pairlist = [k for k in self._exchange.get_markets(quote_currencies=[self._stake_currency], tradable_only=True, active_only=True).keys()]
            _pairlist = self.verify_blacklist(_pairlist, logger.info)
            if not self._use_range:
                filtered_tickers = [v for (k, v) in tickers.items() if self._exchange.get_pair_quote_currency(k) == self._stake_currency and (self._use_range or v.get(self._sort_key) is not None) and (v['symbol'] in _pairlist)]
                pairlist = [s['symbol'] for s in filtered_tickers]
            else:
                pairlist = _pairlist
            pairlist = self.filter_pairlist(pairlist, tickers)
            self._pair_cache['pairlist'] = pairlist.copy()
        return pairlist

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        if False:
            print('Hello World!')
        '\n        Filters and sorts pairlist and returns the whitelist again.\n        Called on each bot iteration - please use internal caching if necessary\n        :param pairlist: pairlist to filter or sort\n        :param tickers: Tickers (from exchange.get_tickers). May be cached.\n        :return: new whitelist\n        '
        if self._use_range:
            filtered_tickers: List[Dict[str, Any]] = [{'symbol': k} for k in pairlist]
            since_ms = int(timeframe_to_prev_date(self._lookback_timeframe, dt_now() + timedelta(minutes=-(self._lookback_period * self._tf_in_min) - self._tf_in_min)).timestamp()) * 1000
            to_ms = int(timeframe_to_prev_date(self._lookback_timeframe, dt_now() - timedelta(minutes=self._tf_in_min)).timestamp()) * 1000
            self.log_once(f'Using volume range of {self._lookback_period} candles, timeframe: {self._lookback_timeframe}, starting from {format_ms_time(since_ms)} till {format_ms_time(to_ms)}', logger.info)
            needed_pairs: ListPairsWithTimeframes = [(p, self._lookback_timeframe, self._def_candletype) for p in [s['symbol'] for s in filtered_tickers] if p not in self._pair_cache]
            candles = {}
            if needed_pairs:
                candles = self._exchange.refresh_latest_ohlcv(needed_pairs, since_ms=since_ms, cache=False)
            for (i, p) in enumerate(filtered_tickers):
                contract_size = self._exchange.markets[p['symbol']].get('contractSize', 1.0) or 1.0
                pair_candles = candles[p['symbol'], self._lookback_timeframe, self._def_candletype] if (p['symbol'], self._lookback_timeframe, self._def_candletype) in candles else None
                if pair_candles is not None and (not pair_candles.empty):
                    if self._exchange.get_option('ohlcv_volume_currency') == 'base':
                        pair_candles['typical_price'] = (pair_candles['high'] + pair_candles['low'] + pair_candles['close']) / 3
                        pair_candles['quoteVolume'] = pair_candles['volume'] * pair_candles['typical_price'] * contract_size
                    else:
                        pair_candles['quoteVolume'] = pair_candles['volume']
                    quoteVolume = pair_candles['quoteVolume'].rolling(self._lookback_period).sum().fillna(0).iloc[-1]
                    filtered_tickers[i]['quoteVolume'] = quoteVolume
                else:
                    filtered_tickers[i]['quoteVolume'] = 0
        else:
            filtered_tickers = [v for (k, v) in tickers.items() if k in pairlist]
        if self._min_value > 0:
            filtered_tickers = [v for v in filtered_tickers if v[self._sort_key] > self._min_value]
        sorted_tickers = sorted(filtered_tickers, reverse=True, key=lambda t: t[self._sort_key])
        pairs = self._whitelist_for_active_markets([s['symbol'] for s in sorted_tickers])
        pairs = self.verify_blacklist(pairs, logmethod=logger.info)
        pairs = pairs[:self._number_pairs]
        return pairs