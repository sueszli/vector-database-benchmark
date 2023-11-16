"""
Cryptocurrency Exchanges support
"""
import asyncio
import inspect
import logging
import signal
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from math import floor
from threading import Lock
from typing import Any, Coroutine, Dict, List, Literal, Optional, Tuple, Union
import ccxt
import ccxt.async_support as ccxt_async
from cachetools import TTLCache
from ccxt import TICK_SIZE
from dateutil import parser
from pandas import DataFrame, concat
from freqtrade.constants import DEFAULT_AMOUNT_RESERVE_PERCENT, NON_OPEN_EXCHANGE_STATES, BidAsk, BuySell, Config, EntryExit, ExchangeConfig, ListPairsWithTimeframes, MakerTaker, OBLiteral, PairWithTimeframe
from freqtrade.data.converter import clean_ohlcv_dataframe, ohlcv_to_dataframe, trades_dict_to_list
from freqtrade.enums import OPTIMIZE_MODES, CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, ExchangeError, InsufficientFundsError, InvalidOrderException, OperationalException, PricingError, RetryableOrderError, TemporaryError
from freqtrade.exchange.common import API_FETCH_ORDER_RETRY_COUNT, remove_exchange_credentials, retrier, retrier_async
from freqtrade.exchange.exchange_utils import ROUND, ROUND_DOWN, ROUND_UP, CcxtModuleType, amount_to_contract_precision, amount_to_contracts, amount_to_precision, contracts_to_amount, date_minus_candles, is_exchange_known_ccxt, market_is_active, price_to_precision, timeframe_to_minutes, timeframe_to_msecs, timeframe_to_next_date, timeframe_to_prev_date, timeframe_to_seconds
from freqtrade.exchange.types import OHLCVResponse, OrderBook, Ticker, Tickers
from freqtrade.misc import chunks, deep_merge_dicts, file_dump_json, file_load_json, safe_value_fallback2
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.util import dt_from_ts, dt_now
from freqtrade.util.datetime_helpers import dt_humanize, dt_ts
logger = logging.getLogger(__name__)

class Exchange:
    _params: Dict = {}
    _ccxt_params: Dict = {}
    _ft_has_default: Dict = {'stoploss_on_exchange': False, 'stop_price_param': 'stopLossPrice', 'stop_price_prop': 'stopLossPrice', 'order_time_in_force': ['GTC'], 'ohlcv_params': {}, 'ohlcv_candle_limit': 500, 'ohlcv_has_history': True, 'ohlcv_partial_candle': True, 'ohlcv_require_since': False, 'ohlcv_volume_currency': 'base', 'tickers_have_quoteVolume': True, 'tickers_have_bid_ask': True, 'tickers_have_price': True, 'trades_pagination': 'time', 'trades_pagination_arg': 'since', 'l2_limit_range': None, 'l2_limit_range_required': True, 'mark_ohlcv_price': 'mark', 'mark_ohlcv_timeframe': '8h', 'ccxt_futures_name': 'swap', 'needs_trading_fees': False, 'order_props_in_contracts': ['amount', 'filled', 'remaining'], 'marketOrderRequiresPrice': False}
    _ft_has: Dict = {}
    _ft_has_futures: Dict = {}
    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = []

    def __init__(self, config: Config, *, exchange_config: Optional[ExchangeConfig]=None, validate: bool=True, load_leverage_tiers: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes this module with the given config,\n        it does basic validation whether the specified exchange and pairs are valid.\n        :return: None\n        '
        self._api: ccxt.Exchange
        self._api_async: ccxt_async.Exchange = None
        self._markets: Dict = {}
        self._trading_fees: Dict[str, Any] = {}
        self._leverage_tiers: Dict[str, List[Dict]] = {}
        self._loop_lock = Lock()
        self.loop = self._init_async_loop()
        self._config: Config = {}
        self._config.update(config)
        self._pairs_last_refresh_time: Dict[PairWithTimeframe, int] = {}
        self._last_markets_refresh: int = 0
        self._cache_lock = Lock()
        self._fetch_tickers_cache: TTLCache = TTLCache(maxsize=2, ttl=60 * 10)
        self._exit_rate_cache: TTLCache = TTLCache(maxsize=100, ttl=1800)
        self._entry_rate_cache: TTLCache = TTLCache(maxsize=100, ttl=1800)
        self._klines: Dict[PairWithTimeframe, DataFrame] = {}
        self._dry_run_open_orders: Dict[str, Any] = {}
        if config['dry_run']:
            logger.info('Instance is running with dry_run enabled')
        logger.info(f'Using CCXT {ccxt.__version__}')
        exchange_conf: Dict[str, Any] = exchange_config if exchange_config else config['exchange']
        remove_exchange_credentials(exchange_conf, config.get('dry_run', False))
        self.log_responses = exchange_conf.get('log_responses', False)
        self.trading_mode: TradingMode = config.get('trading_mode', TradingMode.SPOT)
        self.margin_mode: MarginMode = MarginMode(config.get('margin_mode')) if config.get('margin_mode') else MarginMode.NONE
        self.liquidation_buffer = config.get('liquidation_buffer', 0.05)
        self._ft_has = deep_merge_dicts(self._ft_has, deepcopy(self._ft_has_default))
        if self.trading_mode == TradingMode.FUTURES:
            self._ft_has = deep_merge_dicts(self._ft_has_futures, self._ft_has)
        if exchange_conf.get('_ft_has_params'):
            self._ft_has = deep_merge_dicts(exchange_conf.get('_ft_has_params'), self._ft_has)
            logger.info('Overriding exchange._ft_has with config params, result: %s', self._ft_has)
        self._ohlcv_partial_candle = self._ft_has['ohlcv_partial_candle']
        self._trades_pagination = self._ft_has['trades_pagination']
        self._trades_pagination_arg = self._ft_has['trades_pagination_arg']
        ccxt_config = self._ccxt_config
        ccxt_config = deep_merge_dicts(exchange_conf.get('ccxt_config', {}), ccxt_config)
        ccxt_config = deep_merge_dicts(exchange_conf.get('ccxt_sync_config', {}), ccxt_config)
        self._api = self._init_ccxt(exchange_conf, ccxt_kwargs=ccxt_config)
        ccxt_async_config = self._ccxt_config
        ccxt_async_config = deep_merge_dicts(exchange_conf.get('ccxt_config', {}), ccxt_async_config)
        ccxt_async_config = deep_merge_dicts(exchange_conf.get('ccxt_async_config', {}), ccxt_async_config)
        self._api_async = self._init_ccxt(exchange_conf, ccxt_async, ccxt_kwargs=ccxt_async_config)
        logger.info(f'Using Exchange "{self.name}"')
        self.required_candle_call_count = 1
        if validate:
            self._load_markets()
            self.validate_config(config)
            self._startup_candle_count: int = config.get('startup_candle_count', 0)
            self.required_candle_call_count = self.validate_required_startup_candles(self._startup_candle_count, config.get('timeframe', ''))
        self.markets_refresh_interval: int = exchange_conf.get('markets_refresh_interval', 60) * 60 * 1000
        if self.trading_mode != TradingMode.SPOT and load_leverage_tiers:
            self.fill_leverage_tiers()
        self.additional_exchange_init()

    def __del__(self):
        if False:
            print('Hello World!')
        '\n        Destructor - clean up async stuff\n        '
        self.close()

    def close(self):
        if False:
            i = 10
            return i + 15
        logger.debug('Exchange object destroyed, closing async loop')
        if self._api_async and inspect.iscoroutinefunction(self._api_async.close) and self._api_async.session:
            logger.debug('Closing async ccxt session.')
            self.loop.run_until_complete(self._api_async.close())
        if self.loop and (not self.loop.is_closed()):
            self.loop.close()

    def _init_async_loop(self) -> asyncio.AbstractEventLoop:
        if False:
            i = 10
            return i + 15
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    def validate_config(self, config):
        if False:
            return 10
        self.validate_timeframes(config.get('timeframe'))
        self.validate_stakecurrency(config['stake_currency'])
        if not config['exchange'].get('skip_pair_validation'):
            self.validate_pairs(config['exchange']['pair_whitelist'])
        self.validate_ordertypes(config.get('order_types', {}))
        self.validate_order_time_in_force(config.get('order_time_in_force', {}))
        self.validate_trading_mode_and_margin_mode(self.trading_mode, self.margin_mode)
        self.validate_pricing(config['exit_pricing'])
        self.validate_pricing(config['entry_pricing'])

    def _init_ccxt(self, exchange_config: Dict[str, Any], ccxt_module: CcxtModuleType=ccxt, ccxt_kwargs: Dict={}) -> ccxt.Exchange:
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize ccxt with given config and return valid\n        ccxt instance.\n        '
        name = exchange_config['name']
        if not is_exchange_known_ccxt(name, ccxt_module):
            raise OperationalException(f'Exchange {name} is not supported by ccxt')
        ex_config = {'apiKey': exchange_config.get('key'), 'secret': exchange_config.get('secret'), 'password': exchange_config.get('password'), 'uid': exchange_config.get('uid', '')}
        if ccxt_kwargs:
            logger.info('Applying additional ccxt config: %s', ccxt_kwargs)
        if self._ccxt_params:
            ccxt_kwargs = deep_merge_dicts(self._ccxt_params, ccxt_kwargs)
        if ccxt_kwargs:
            ex_config.update(ccxt_kwargs)
        try:
            api = getattr(ccxt_module, name.lower())(ex_config)
        except (KeyError, AttributeError) as e:
            raise OperationalException(f'Exchange {name} is not supported') from e
        except ccxt.BaseError as e:
            raise OperationalException(f'Initialization of ccxt failed. Reason: {e}') from e
        return api

    @property
    def _ccxt_config(self) -> Dict:
        if False:
            i = 10
            return i + 15
        if self.trading_mode == TradingMode.MARGIN:
            return {'options': {'defaultType': 'margin'}}
        elif self.trading_mode == TradingMode.FUTURES:
            return {'options': {'defaultType': self._ft_has['ccxt_futures_name']}}
        else:
            return {}

    @property
    def name(self) -> str:
        if False:
            return 10
        'exchange Name (from ccxt)'
        return self._api.name

    @property
    def id(self) -> str:
        if False:
            i = 10
            return i + 15
        'exchange ccxt id'
        return self._api.id

    @property
    def timeframes(self) -> List[str]:
        if False:
            return 10
        return list((self._api.timeframes or {}).keys())

    @property
    def markets(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'exchange ccxt markets'
        if not self._markets:
            logger.info('Markets were not loaded. Loading them now..')
            self._load_markets()
        return self._markets

    @property
    def precisionMode(self) -> int:
        if False:
            i = 10
            return i + 15
        'exchange ccxt precisionMode'
        return self._api.precisionMode

    def additional_exchange_init(self) -> None:
        if False:
            return 10
        '\n        Additional exchange initialization logic.\n        .api will be available at this point.\n        Must be overridden in child methods if required.\n        '
        pass

    def _log_exchange_response(self, endpoint, response) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Log exchange responses '
        if self.log_responses:
            logger.info(f'API {endpoint}: {response}')

    def ohlcv_candle_limit(self, timeframe: str, candle_type: CandleType, since_ms: Optional[int]=None) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Exchange ohlcv candle limit\n        Uses ohlcv_candle_limit_per_timeframe if the exchange has different limits\n        per timeframe (e.g. bittrex), otherwise falls back to ohlcv_candle_limit\n        :param timeframe: Timeframe to check\n        :param candle_type: Candle-type\n        :param since_ms: Starting timestamp\n        :return: Candle limit as integer\n        '
        return int(self._ft_has.get('ohlcv_candle_limit_per_timeframe', {}).get(timeframe, self._ft_has.get('ohlcv_candle_limit')))

    def get_markets(self, base_currencies: List[str]=[], quote_currencies: List[str]=[], spot_only: bool=False, margin_only: bool=False, futures_only: bool=False, tradable_only: bool=True, active_only: bool=False) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Return exchange ccxt markets, filtered out by base currency and quote currency\n        if this was requested in parameters.\n        '
        markets = self.markets
        if not markets:
            raise OperationalException('Markets were not loaded.')
        if base_currencies:
            markets = {k: v for (k, v) in markets.items() if v['base'] in base_currencies}
        if quote_currencies:
            markets = {k: v for (k, v) in markets.items() if v['quote'] in quote_currencies}
        if tradable_only:
            markets = {k: v for (k, v) in markets.items() if self.market_is_tradable(v)}
        if spot_only:
            markets = {k: v for (k, v) in markets.items() if self.market_is_spot(v)}
        if margin_only:
            markets = {k: v for (k, v) in markets.items() if self.market_is_margin(v)}
        if futures_only:
            markets = {k: v for (k, v) in markets.items() if self.market_is_future(v)}
        if active_only:
            markets = {k: v for (k, v) in markets.items() if market_is_active(v)}
        return markets

    def get_quote_currencies(self) -> List[str]:
        if False:
            print('Hello World!')
        '\n        Return a list of supported quote currencies\n        '
        markets = self.markets
        return sorted(set([x['quote'] for (_, x) in markets.items()]))

    def get_pair_quote_currency(self, pair: str) -> str:
        if False:
            print('Hello World!')
        " Return a pair's quote currency (base/quote:settlement) "
        return self.markets.get(pair, {}).get('quote', '')

    def get_pair_base_currency(self, pair: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        " Return a pair's base currency (base/quote:settlement) "
        return self.markets.get(pair, {}).get('base', '')

    def market_is_future(self, market: Dict[str, Any]) -> bool:
        if False:
            i = 10
            return i + 15
        return market.get(self._ft_has['ccxt_futures_name'], False) is True and market.get('linear', False) is True

    def market_is_spot(self, market: Dict[str, Any]) -> bool:
        if False:
            i = 10
            return i + 15
        return market.get('spot', False) is True

    def market_is_margin(self, market: Dict[str, Any]) -> bool:
        if False:
            print('Hello World!')
        return market.get('margin', False) is True

    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        if False:
            print('Hello World!')
        '\n        Check if the market symbol is tradable by Freqtrade.\n        Ensures that Configured mode aligns to\n        '
        return market.get('quote', None) is not None and market.get('base', None) is not None and (self.precisionMode != TICK_SIZE or market.get('precision', {}).get('price') > 1e-11) and (self.trading_mode == TradingMode.SPOT and self.market_is_spot(market) or (self.trading_mode == TradingMode.MARGIN and self.market_is_margin(market)) or (self.trading_mode == TradingMode.FUTURES and self.market_is_future(market)))

    def klines(self, pair_interval: PairWithTimeframe, copy: bool=True) -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        if pair_interval in self._klines:
            return self._klines[pair_interval].copy() if copy else self._klines[pair_interval]
        else:
            return DataFrame()

    def get_contract_size(self, pair: str) -> Optional[float]:
        if False:
            i = 10
            return i + 15
        if self.trading_mode == TradingMode.FUTURES:
            market = self.markets.get(pair, {})
            contract_size: float = 1.0
            if not market:
                return None
            if market.get('contractSize') is not None:
                contract_size = float(market['contractSize'])
            return contract_size
        else:
            return 1

    def _trades_contracts_to_amount(self, trades: List) -> List:
        if False:
            while True:
                i = 10
        if len(trades) > 0 and 'symbol' in trades[0]:
            contract_size = self.get_contract_size(trades[0]['symbol'])
            if contract_size != 1:
                for trade in trades:
                    trade['amount'] = trade['amount'] * contract_size
        return trades

    def _order_contracts_to_amount(self, order: Dict) -> Dict:
        if False:
            print('Hello World!')
        if 'symbol' in order and order['symbol'] is not None:
            contract_size = self.get_contract_size(order['symbol'])
            if contract_size != 1:
                for prop in self._ft_has.get('order_props_in_contracts', []):
                    if prop in order and order[prop] is not None:
                        order[prop] = order[prop] * contract_size
        return order

    def _amount_to_contracts(self, pair: str, amount: float) -> float:
        if False:
            for i in range(10):
                print('nop')
        contract_size = self.get_contract_size(pair)
        return amount_to_contracts(amount, contract_size)

    def _contracts_to_amount(self, pair: str, num_contracts: float) -> float:
        if False:
            while True:
                i = 10
        contract_size = self.get_contract_size(pair)
        return contracts_to_amount(num_contracts, contract_size)

    def amount_to_contract_precision(self, pair: str, amount: float) -> float:
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper wrapper around amount_to_contract_precision\n        '
        contract_size = self.get_contract_size(pair)
        return amount_to_contract_precision(amount, self.get_precision_amount(pair), self.precisionMode, contract_size)

    def _load_async_markets(self, reload: bool=False) -> None:
        if False:
            return 10
        try:
            if self._api_async:
                self.loop.run_until_complete(self._api_async.load_markets(reload=reload, params={}))
        except (asyncio.TimeoutError, ccxt.BaseError) as e:
            logger.warning('Could not load async markets. Reason: %s', e)
            return

    def _load_markets(self) -> None:
        if False:
            while True:
                i = 10
        ' Initialize markets both sync and async '
        try:
            self._markets = self._api.load_markets(params={})
            self._load_async_markets()
            self._last_markets_refresh = dt_ts()
            if self._ft_has['needs_trading_fees']:
                self._trading_fees = self.fetch_trading_fees()
        except ccxt.BaseError:
            logger.exception('Unable to initialize markets.')

    def reload_markets(self) -> None:
        if False:
            print('Hello World!')
        'Reload markets both sync and async if refresh interval has passed '
        if self._last_markets_refresh > 0 and self._last_markets_refresh + self.markets_refresh_interval > dt_ts():
            return None
        logger.debug('Performing scheduled market reload..')
        try:
            self._markets = self._api.load_markets(reload=True, params={})
            self._load_async_markets(reload=True)
            self._last_markets_refresh = dt_ts()
            self.fill_leverage_tiers()
        except ccxt.BaseError:
            logger.exception('Could not reload markets.')

    def validate_stakecurrency(self, stake_currency: str) -> None:
        if False:
            while True:
                i = 10
        "\n        Checks stake-currency against available currencies on the exchange.\n        Only runs on startup. If markets have not been loaded, there's been a problem with\n        the connection to the exchange.\n        :param stake_currency: Stake-currency to validate\n        :raise: OperationalException if stake-currency is not available.\n        "
        if not self._markets:
            raise OperationalException('Could not load markets, therefore cannot start. Please investigate the above error for more details.')
        quote_currencies = self.get_quote_currencies()
        if stake_currency not in quote_currencies:
            raise OperationalException(f"{stake_currency} is not available as stake on {self.name}. Available currencies are: {', '.join(quote_currencies)}")

    def validate_pairs(self, pairs: List[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if all given pairs are tradable on the current exchange.\n        :param pairs: list of pairs\n        :raise: OperationalException if one pair is not available\n        :return: None\n        '
        if not self.markets:
            logger.warning('Unable to validate pairs (assuming they are correct).')
            return
        extended_pairs = expand_pairlist(pairs, list(self.markets), keep_invalid=True)
        invalid_pairs = []
        for pair in extended_pairs:
            if self.markets and pair not in self.markets:
                raise OperationalException(f'Pair {pair} is not available on {self.name} {self.trading_mode.value}. Please remove {pair} from your whitelist.')
            elif isinstance(self.markets[pair].get('info'), dict) and self.markets[pair].get('info', {}).get('prohibitedIn', False):
                logger.warning(f'Pair {pair} is restricted for some users on this exchange.Please check if you are impacted by this restriction on the exchange and eventually remove {pair} from your whitelist.')
            if self._config['stake_currency'] and self.get_pair_quote_currency(pair) != self._config['stake_currency']:
                invalid_pairs.append(pair)
        if invalid_pairs:
            raise OperationalException(f"Stake-currency '{self._config['stake_currency']}' not compatible with pair-whitelist. Please remove the following pairs: {invalid_pairs}")

    def get_valid_pair_combination(self, curr_1: str, curr_2: str) -> str:
        if False:
            return 10
        '\n        Get valid pair combination of curr_1 and curr_2 by trying both combinations.\n        '
        for pair in [f'{curr_1}/{curr_2}', f'{curr_2}/{curr_1}']:
            if pair in self.markets and self.markets[pair].get('active'):
                return pair
        raise ValueError(f'Could not combine {curr_1} and {curr_2} to get a valid pair.')

    def validate_timeframes(self, timeframe: Optional[str]) -> None:
        if False:
            return 10
        '\n        Check if timeframe from config is a supported timeframe on the exchange\n        '
        if not hasattr(self._api, 'timeframes') or self._api.timeframes is None:
            raise OperationalException(f"The ccxt library does not provide the list of timeframes for the exchange {self.name} and this exchange is therefore not supported. ccxt fetchOHLCV: {self.exchange_has('fetchOHLCV')}")
        if timeframe and timeframe not in self.timeframes:
            raise OperationalException(f"Invalid timeframe '{timeframe}'. This exchange supports: {self.timeframes}")
        if timeframe and timeframe_to_minutes(timeframe) < 1:
            raise OperationalException('Timeframes < 1m are currently not supported by Freqtrade.')

    def validate_ordertypes(self, order_types: Dict) -> None:
        if False:
            print('Hello World!')
        '\n        Checks if order-types configured in strategy/config are supported\n        '
        if any((v == 'market' for (k, v) in order_types.items())):
            if not self.exchange_has('createMarketOrder'):
                raise OperationalException(f'Exchange {self.name} does not support market orders.')
        self.validate_stop_ordertypes(order_types)

    def validate_stop_ordertypes(self, order_types: Dict) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Validate stoploss order types\n        '
        if order_types.get('stoploss_on_exchange') and (not self._ft_has.get('stoploss_on_exchange', False)):
            raise OperationalException(f'On exchange stoploss is not supported for {self.name}.')
        if self.trading_mode == TradingMode.FUTURES:
            price_mapping = self._ft_has.get('stop_price_type_value_mapping', {}).keys()
            if order_types.get('stoploss_on_exchange', False) is True and 'stoploss_price_type' in order_types and (order_types['stoploss_price_type'] not in price_mapping):
                raise OperationalException(f'On exchange stoploss price type is not supported for {self.name}.')

    def validate_pricing(self, pricing: Dict) -> None:
        if False:
            return 10
        if pricing.get('use_order_book', False) and (not self.exchange_has('fetchL2OrderBook')):
            raise OperationalException(f'Orderbook not available for {self.name}.')
        if not pricing.get('use_order_book', False) and (not self.exchange_has('fetchTicker') or not self._ft_has['tickers_have_price']):
            raise OperationalException(f'Ticker pricing not available for {self.name}.')

    def validate_order_time_in_force(self, order_time_in_force: Dict) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Checks if order time in force configured in strategy/config are supported\n        '
        if any((v.upper() not in self._ft_has['order_time_in_force'] for (k, v) in order_time_in_force.items())):
            raise OperationalException(f'Time in force policies are not supported for {self.name} yet.')

    def validate_required_startup_candles(self, startup_candles: int, timeframe: str) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Checks if required startup_candles is more than ohlcv_candle_limit().\n        Requires a grace-period of 5 candles - so a startup-period up to 494 is allowed by default.\n        '
        candle_limit = self.ohlcv_candle_limit(timeframe, self._config['candle_type_def'], int(date_minus_candles(timeframe, startup_candles).timestamp() * 1000) if timeframe else None)
        candle_count = startup_candles + 1
        required_candle_call_count = int(candle_count / candle_limit + (0 if candle_count % candle_limit == 0 else 1))
        if self._ft_has['ohlcv_has_history']:
            if required_candle_call_count > 5:
                raise OperationalException(f'This strategy requires {startup_candles} candles to start, which is more than 5x the amount of candles {self.name} provides for {timeframe}.')
        elif required_candle_call_count > 1:
            raise OperationalException(f'This strategy requires {startup_candles} candles to start, which is more than the amount of candles {self.name} provides for {timeframe}.')
        if required_candle_call_count > 1:
            logger.warning(f'Using {required_candle_call_count} calls to get OHLCV. This can result in slower operations for the bot. Please check if you really need {startup_candles} candles for your strategy')
        return required_candle_call_count

    def validate_trading_mode_and_margin_mode(self, trading_mode: TradingMode, margin_mode: Optional[MarginMode]):
        if False:
            while True:
                i = 10
        '\n        Checks if freqtrade can perform trades using the configured\n        trading mode(Margin, Futures) and MarginMode(Cross, Isolated)\n        Throws OperationalException:\n            If the trading_mode/margin_mode type are not supported by freqtrade on this exchange\n        '
        if trading_mode != TradingMode.SPOT and (trading_mode, margin_mode) not in self._supported_trading_mode_margin_pairs:
            mm_value = margin_mode and margin_mode.value
            raise OperationalException(f'Freqtrade does not support {mm_value} {trading_mode.value} on {self.name}')

    def get_option(self, param: str, default: Optional[Any]=None) -> Any:
        if False:
            while True:
                i = 10
        '\n        Get parameter value from _ft_has\n        '
        return self._ft_has.get(param, default)

    def exchange_has(self, endpoint: str) -> bool:
        if False:
            while True:
                i = 10
        "\n        Checks if exchange implements a specific API endpoint.\n        Wrapper around ccxt 'has' attribute\n        :param endpoint: Name of endpoint (e.g. 'fetchOHLCV', 'fetchTickers')\n        :return: bool\n        "
        return endpoint in self._api.has and self._api.has[endpoint]

    def get_precision_amount(self, pair: str) -> Optional[float]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the amount precision of the exchange.\n        :param pair: Pair to get precision for\n        :return: precision for amount or None. Must be used in combination with precisionMode\n        '
        return self.markets.get(pair, {}).get('precision', {}).get('amount', None)

    def get_precision_price(self, pair: str) -> Optional[float]:
        if False:
            i = 10
            return i + 15
        '\n        Returns the price precision of the exchange.\n        :param pair: Pair to get precision for\n        :return: precision for price or None. Must be used in combination with precisionMode\n        '
        return self.markets.get(pair, {}).get('precision', {}).get('price', None)

    def amount_to_precision(self, pair: str, amount: float) -> float:
        if False:
            i = 10
            return i + 15
        '\n        Returns the amount to buy or sell to a precision the Exchange accepts\n\n        '
        return amount_to_precision(amount, self.get_precision_amount(pair), self.precisionMode)

    def price_to_precision(self, pair: str, price: float, *, rounding_mode: int=ROUND) -> float:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the price rounded to the precision the Exchange accepts.\n        The default price_rounding_mode in conf is ROUND.\n        For stoploss calculations, must use ROUND_UP for longs, and ROUND_DOWN for shorts.\n        '
        return price_to_precision(price, self.get_precision_price(pair), self.precisionMode, rounding_mode=rounding_mode)

    def price_get_one_pip(self, pair: str, price: float) -> float:
        if False:
            while True:
                i = 10
        '\n        Get\'s the "1 pip" value for this pair.\n        Used in PriceFilter to calculate the 1pip movements.\n        '
        precision = self.markets[pair]['precision']['price']
        if self.precisionMode == TICK_SIZE:
            return precision
        else:
            return 1 / pow(10, precision)

    def get_min_pair_stake_amount(self, pair: str, price: float, stoploss: float, leverage: Optional[float]=1.0) -> Optional[float]:
        if False:
            i = 10
            return i + 15
        return self._get_stake_amount_limit(pair, price, stoploss, 'min', leverage)

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float=1.0) -> float:
        if False:
            while True:
                i = 10
        max_stake_amount = self._get_stake_amount_limit(pair, price, 0.0, 'max', leverage)
        if max_stake_amount is None:
            raise OperationalException(f'{self.name}.get_max_pair_stake_amount shouldnever set max_stake_amount to None')
        return max_stake_amount

    def _get_stake_amount_limit(self, pair: str, price: float, stoploss: float, limit: Literal['min', 'max'], leverage: Optional[float]=1.0) -> Optional[float]:
        if False:
            while True:
                i = 10
        isMin = limit == 'min'
        try:
            market = self.markets[pair]
        except KeyError:
            raise ValueError(f"Can't get market information for symbol {pair}")
        if isMin:
            margin_reserve: float = 1.0 + self._config.get('amount_reserve_percent', DEFAULT_AMOUNT_RESERVE_PERCENT)
            stoploss_reserve = margin_reserve / (1 - abs(stoploss)) if abs(stoploss) != 1 else 1.5
            stoploss_reserve = max(min(stoploss_reserve, 1.5), 1)
        else:
            margin_reserve = 1.0
            stoploss_reserve = 1.0
        stake_limits = []
        limits = market['limits']
        if limits['cost'][limit] is not None:
            stake_limits.append(self._contracts_to_amount(pair, limits['cost'][limit]) * stoploss_reserve)
        if limits['amount'][limit] is not None:
            stake_limits.append(self._contracts_to_amount(pair, limits['amount'][limit]) * price * margin_reserve)
        if not stake_limits:
            return None if isMin else float('inf')
        return self._get_stake_amount_considering_leverage(max(stake_limits) if isMin else min(stake_limits), leverage or 1.0)

    def _get_stake_amount_considering_leverage(self, stake_amount: float, leverage: float) -> float:
        if False:
            return 10
        '\n        Takes the minimum stake amount for a pair with no leverage and returns the minimum\n        stake amount when leverage is considered\n        :param stake_amount: The stake amount for a pair before leverage is considered\n        :param leverage: The amount of leverage being used on the current trade\n        '
        return stake_amount / leverage

    def create_dry_run_order(self, pair: str, ordertype: str, side: str, amount: float, rate: float, leverage: float, params: Dict={}, stop_loss: bool=False) -> Dict[str, Any]:
        if False:
            return 10
        now = dt_now()
        order_id = f'dry_run_{side}_{pair}_{now.timestamp()}'
        _amount = self._contracts_to_amount(pair, self.amount_to_precision(pair, self._amount_to_contracts(pair, amount)))
        dry_order: Dict[str, Any] = {'id': order_id, 'symbol': pair, 'price': rate, 'average': rate, 'amount': _amount, 'cost': _amount * rate, 'type': ordertype, 'side': side, 'filled': 0, 'remaining': _amount, 'datetime': now.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'timestamp': dt_ts(now), 'status': 'open', 'fee': None, 'info': {}, 'leverage': leverage}
        if stop_loss:
            dry_order['info'] = {'stopPrice': dry_order['price']}
            dry_order[self._ft_has['stop_price_prop']] = dry_order['price']
            dry_order['ft_order_type'] = 'stoploss'
        orderbook: Optional[OrderBook] = None
        if self.exchange_has('fetchL2OrderBook'):
            orderbook = self.fetch_l2_order_book(pair, 20)
        if ordertype == 'limit' and orderbook:
            allowed_diff = 0.01
            if self._dry_is_price_crossed(pair, side, rate, orderbook, allowed_diff):
                logger.info(f'Converted order {pair} to market order due to price {rate} crossing spread by more than {allowed_diff:.2%}.')
                dry_order['type'] = 'market'
        if dry_order['type'] == 'market' and (not dry_order.get('ft_order_type')):
            average = self.get_dry_market_fill_price(pair, side, amount, rate, orderbook)
            dry_order.update({'average': average, 'filled': _amount, 'remaining': 0.0, 'status': 'closed', 'cost': dry_order['amount'] * average})
            dry_order = self.add_dry_order_fee(pair, dry_order, 'taker')
        dry_order = self.check_dry_limit_order_filled(dry_order, immediate=True, orderbook=orderbook)
        self._dry_run_open_orders[dry_order['id']] = dry_order
        return dry_order

    def add_dry_order_fee(self, pair: str, dry_order: Dict[str, Any], taker_or_maker: MakerTaker) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        fee = self.get_fee(pair, taker_or_maker=taker_or_maker)
        dry_order.update({'fee': {'currency': self.get_pair_quote_currency(pair), 'cost': dry_order['cost'] * fee, 'rate': fee}})
        return dry_order

    def get_dry_market_fill_price(self, pair: str, side: str, amount: float, rate: float, orderbook: Optional[OrderBook]) -> float:
        if False:
            while True:
                i = 10
        '\n        Get the market order fill price based on orderbook interpolation\n        '
        if self.exchange_has('fetchL2OrderBook'):
            if not orderbook:
                orderbook = self.fetch_l2_order_book(pair, 20)
            ob_type: OBLiteral = 'asks' if side == 'buy' else 'bids'
            slippage = 0.05
            max_slippage_val = rate * (1 + slippage if side == 'buy' else 1 - slippage)
            remaining_amount = amount
            filled_value = 0.0
            book_entry_price = 0.0
            for book_entry in orderbook[ob_type]:
                book_entry_price = book_entry[0]
                book_entry_coin_volume = book_entry[1]
                if remaining_amount > 0:
                    if remaining_amount < book_entry_coin_volume:
                        filled_value += remaining_amount * book_entry_price
                        break
                    else:
                        filled_value += book_entry_coin_volume * book_entry_price
                    remaining_amount -= book_entry_coin_volume
                else:
                    break
            else:
                filled_value += remaining_amount * book_entry_price
            forecast_avg_filled_price = max(filled_value, 0) / amount
            if side == 'buy':
                forecast_avg_filled_price = min(forecast_avg_filled_price, max_slippage_val)
            else:
                forecast_avg_filled_price = max(forecast_avg_filled_price, max_slippage_val)
            return self.price_to_precision(pair, forecast_avg_filled_price)
        return rate

    def _dry_is_price_crossed(self, pair: str, side: str, limit: float, orderbook: Optional[OrderBook]=None, offset: float=0.0) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not self.exchange_has('fetchL2OrderBook'):
            return True
        if not orderbook:
            orderbook = self.fetch_l2_order_book(pair, 1)
        try:
            if side == 'buy':
                price = orderbook['asks'][0][0]
                if limit * (1 - offset) >= price:
                    return True
            else:
                price = orderbook['bids'][0][0]
                if limit * (1 + offset) <= price:
                    return True
        except IndexError:
            pass
        return False

    def check_dry_limit_order_filled(self, order: Dict[str, Any], immediate: bool=False, orderbook: Optional[OrderBook]=None) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Check dry-run limit order fill and update fee (if it filled).\n        '
        if order['status'] != 'closed' and order['type'] in ['limit'] and (not order.get('ft_order_type')):
            pair = order['symbol']
            if self._dry_is_price_crossed(pair, order['side'], order['price'], orderbook):
                order.update({'status': 'closed', 'filled': order['amount'], 'remaining': 0})
                self.add_dry_order_fee(pair, order, 'taker' if immediate else 'maker')
        return order

    def fetch_dry_run_order(self, order_id) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        Return dry-run order\n        Only call if running in dry-run mode.\n        '
        try:
            order = self._dry_run_open_orders[order_id]
            order = self.check_dry_limit_order_filled(order)
            return order
        except KeyError as e:
            from freqtrade.persistence import Order
            order = Order.order_by_id(order_id)
            if order:
                ccxt_order = order.to_ccxt_object(self._ft_has['stop_price_prop'])
                self._dry_run_open_orders[order_id] = ccxt_order
                return ccxt_order
            raise InvalidOrderException(f'Tried to get an invalid dry-run-order (id: {order_id}). Message: {e}') from e

    def _lev_prep(self, pair: str, leverage: float, side: BuySell, accept_fail: bool=False):
        if False:
            print('Hello World!')
        if self.trading_mode != TradingMode.SPOT:
            self.set_margin_mode(pair, self.margin_mode, accept_fail)
            self._set_leverage(leverage, pair, accept_fail)

    def _get_params(self, side: BuySell, ordertype: str, leverage: float, reduceOnly: bool, time_in_force: str='GTC') -> Dict:
        if False:
            for i in range(10):
                print('nop')
        params = self._params.copy()
        if time_in_force != 'GTC' and ordertype != 'market':
            params.update({'timeInForce': time_in_force.upper()})
        if reduceOnly:
            params.update({'reduceOnly': True})
        return params

    def _order_needs_price(self, ordertype: str) -> bool:
        if False:
            while True:
                i = 10
        return ordertype != 'market' or self._api.options.get('createMarketBuyOrderRequiresPrice', False) or self._ft_has.get('marketOrderRequiresPrice', False)

    def create_order(self, *, pair: str, ordertype: str, side: BuySell, amount: float, rate: float, leverage: float, reduceOnly: bool=False, time_in_force: str='GTC') -> Dict:
        if False:
            return 10
        if self._config['dry_run']:
            dry_order = self.create_dry_run_order(pair, ordertype, side, amount, self.price_to_precision(pair, rate), leverage)
            return dry_order
        params = self._get_params(side, ordertype, leverage, reduceOnly, time_in_force)
        try:
            amount = self.amount_to_precision(pair, self._amount_to_contracts(pair, amount))
            needs_price = self._order_needs_price(ordertype)
            rate_for_order = self.price_to_precision(pair, rate) if needs_price else None
            if not reduceOnly:
                self._lev_prep(pair, leverage, side)
            order = self._api.create_order(pair, ordertype, side, amount, rate_for_order, params)
            if order.get('status') is None:
                order['status'] = 'open'
            if order.get('type') is None:
                order['type'] = ordertype
            self._log_exchange_response('create_order', order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(f'Insufficient funds to create {ordertype} {side} order on market {pair}. Tried to {side} amount {amount} at rate {rate}.Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(f'Could not create {ordertype} {side} order on market {pair}. Tried to {side} amount {amount} at rate {rate}. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not place {side} order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def stoploss_adjust(self, stop_loss: float, order: Dict, side: str) -> bool:
        if False:
            while True:
                i = 10
        '\n        Verify stop_loss against stoploss-order value (limit or price)\n        Returns True if adjustment is necessary.\n        '
        if not self._ft_has.get('stoploss_on_exchange'):
            raise OperationalException(f'stoploss is not implemented for {self.name}.')
        price_param = self._ft_has['stop_price_prop']
        return order.get(price_param, None) is None or (side == 'sell' and stop_loss > float(order[price_param]) or (side == 'buy' and stop_loss < float(order[price_param])))

    def _get_stop_order_type(self, user_order_type) -> Tuple[str, str]:
        if False:
            i = 10
            return i + 15
        available_order_Types: Dict[str, str] = self._ft_has['stoploss_order_types']
        if user_order_type in available_order_Types.keys():
            ordertype = available_order_Types[user_order_type]
        else:
            ordertype = list(available_order_Types.values())[0]
            user_order_type = list(available_order_Types.keys())[0]
        return (ordertype, user_order_type)

    def _get_stop_limit_rate(self, stop_price: float, order_types: Dict, side: str) -> float:
        if False:
            i = 10
            return i + 15
        limit_price_pct = order_types.get('stoploss_on_exchange_limit_ratio', 0.99)
        if side == 'sell':
            limit_rate = stop_price * limit_price_pct
        else:
            limit_rate = stop_price * (2 - limit_price_pct)
        bad_stop_price = stop_price < limit_rate if side == 'sell' else stop_price > limit_rate
        if bad_stop_price:
            raise InvalidOrderException(f'In stoploss limit order, stop price should be more than limit price. Stop price: {stop_price}, Limit price: {limit_rate}, Limit Price pct: {limit_price_pct}')
        return limit_rate

    def _get_stop_params(self, side: BuySell, ordertype: str, stop_price: float) -> Dict:
        if False:
            while True:
                i = 10
        params = self._params.copy()
        params.update({self._ft_has['stop_price_param']: stop_price})
        return params

    @retrier(retries=0)
    def create_stoploss(self, pair: str, amount: float, stop_price: float, order_types: Dict, side: BuySell, leverage: float) -> Dict:
        if False:
            while True:
                i = 10
        "\n        creates a stoploss order.\n        requires `_ft_has['stoploss_order_types']` to be set as a dict mapping limit and market\n            to the corresponding exchange type.\n\n        The precise ordertype is determined by the order_types dict or exchange default.\n\n        The exception below should never raise, since we disallow\n        starting the bot in validate_ordertypes()\n\n        This may work with a limited number of other exchanges, but correct working\n            needs to be tested individually.\n        WARNING: setting `stoploss_on_exchange` to True will NOT auto-enable stoploss on exchange.\n            `stoploss_adjust` must still be implemented for this to work.\n        "
        if not self._ft_has['stoploss_on_exchange']:
            raise OperationalException(f'stoploss is not implemented for {self.name}.')
        user_order_type = order_types.get('stoploss', 'market')
        (ordertype, user_order_type) = self._get_stop_order_type(user_order_type)
        round_mode = ROUND_DOWN if side == 'buy' else ROUND_UP
        stop_price_norm = self.price_to_precision(pair, stop_price, rounding_mode=round_mode)
        limit_rate = None
        if user_order_type == 'limit':
            limit_rate = self._get_stop_limit_rate(stop_price, order_types, side)
            limit_rate = self.price_to_precision(pair, limit_rate, rounding_mode=round_mode)
        if self._config['dry_run']:
            dry_order = self.create_dry_run_order(pair, ordertype, side, amount, stop_price_norm, stop_loss=True, leverage=leverage)
            return dry_order
        try:
            params = self._get_stop_params(side=side, ordertype=ordertype, stop_price=stop_price_norm)
            if self.trading_mode == TradingMode.FUTURES:
                params['reduceOnly'] = True
                if 'stoploss_price_type' in order_types and 'stop_price_type_field' in self._ft_has:
                    price_type = self._ft_has['stop_price_type_value_mapping'][order_types.get('stoploss_price_type', PriceType.LAST)]
                    params[self._ft_has['stop_price_type_field']] = price_type
            amount = self.amount_to_precision(pair, self._amount_to_contracts(pair, amount))
            self._lev_prep(pair, leverage, side, accept_fail=True)
            order = self._api.create_order(symbol=pair, type=ordertype, side=side, amount=amount, price=limit_rate, params=params)
            self._log_exchange_response('create_stoploss_order', order)
            order = self._order_contracts_to_amount(order)
            logger.info(f'stoploss {user_order_type} order added for {pair}. stop price: {stop_price}. limit: {limit_rate}')
            return order
        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(f'Insufficient funds to create {ordertype} sell order on market {pair}. Tried to sell amount {amount} at rate {limit_rate}. Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(f'Could not create {ordertype} sell order on market {pair}. Tried to sell amount {amount} at rate {limit_rate}. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not place stoploss order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier(retries=API_FETCH_ORDER_RETRY_COUNT)
    def fetch_order(self, order_id: str, pair: str, params: Dict={}) -> Dict:
        if False:
            while True:
                i = 10
        if self._config['dry_run']:
            return self.fetch_dry_run_order(order_id)
        try:
            order = self._api.fetch_order(order_id, pair, params=params)
            self._log_exchange_response('fetch_order', order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.OrderNotFound as e:
            raise RetryableOrderError(f'Order not found (pair: {pair} id: {order_id}). Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(f'Tried to get an invalid order (pair: {pair} id: {order_id}). Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def fetch_stoploss_order(self, order_id: str, pair: str, params: Dict={}) -> Dict:
        if False:
            i = 10
            return i + 15
        return self.fetch_order(order_id, pair, params)

    def fetch_order_or_stoploss_order(self, order_id: str, pair: str, stoploss_order: bool=False) -> Dict:
        if False:
            i = 10
            return i + 15
        '\n        Simple wrapper calling either fetch_order or fetch_stoploss_order depending on\n        the stoploss_order parameter\n        :param order_id: OrderId to fetch order\n        :param pair: Pair corresponding to order_id\n        :param stoploss_order: If true, uses fetch_stoploss_order, otherwise fetch_order.\n        '
        if stoploss_order:
            return self.fetch_stoploss_order(order_id, pair)
        return self.fetch_order(order_id, pair)

    def check_order_canceled_empty(self, order: Dict) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify if an order has been cancelled without being partially filled\n        :param order: Order dict as returned from fetch_order()\n        :return: True if order has been cancelled without being filled, False otherwise.\n        '
        return order.get('status') in NON_OPEN_EXCHANGE_STATES and order.get('filled') == 0.0

    @retrier
    def cancel_order(self, order_id: str, pair: str, params: Dict={}) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        if self._config['dry_run']:
            try:
                order = self.fetch_dry_run_order(order_id)
                order.update({'status': 'canceled', 'filled': 0.0, 'remaining': order['amount']})
                return order
            except InvalidOrderException:
                return {}
        try:
            order = self._api.cancel_order(order_id, pair, params=params)
            self._log_exchange_response('cancel_order', order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(f'Could not cancel order. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not cancel order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def cancel_stoploss_order(self, order_id: str, pair: str, params: Dict={}) -> Dict:
        if False:
            print('Hello World!')
        return self.cancel_order(order_id, pair, params)

    def is_cancel_order_result_suitable(self, corder) -> bool:
        if False:
            i = 10
            return i + 15
        if not isinstance(corder, dict):
            return False
        required = ('fee', 'status', 'amount')
        return all((corder.get(k, None) is not None for k in required))

    def cancel_order_with_result(self, order_id: str, pair: str, amount: float) -> Dict:
        if False:
            while True:
                i = 10
        "\n        Cancel order returning a result.\n        Creates a fake result if cancel order returns a non-usable result\n        and fetch_order does not work (certain exchanges don't return cancelled orders)\n        :param order_id: Orderid to cancel\n        :param pair: Pair corresponding to order_id\n        :param amount: Amount to use for fake response\n        :return: Result from either cancel_order if usable, or fetch_order\n        "
        try:
            corder = self.cancel_order(order_id, pair)
            if self.is_cancel_order_result_suitable(corder):
                return corder
        except InvalidOrderException:
            logger.warning(f'Could not cancel order {order_id} for {pair}.')
        try:
            order = self.fetch_order(order_id, pair)
        except InvalidOrderException:
            logger.warning(f'Could not fetch cancelled order {order_id}.')
            order = {'id': order_id, 'status': 'canceled', 'amount': amount, 'filled': 0.0, 'fee': {}, 'info': {}}
        return order

    def cancel_stoploss_order_with_result(self, order_id: str, pair: str, amount: float) -> Dict:
        if False:
            i = 10
            return i + 15
        "\n        Cancel stoploss order returning a result.\n        Creates a fake result if cancel order returns a non-usable result\n        and fetch_order does not work (certain exchanges don't return cancelled orders)\n        :param order_id: stoploss-order-id to cancel\n        :param pair: Pair corresponding to order_id\n        :param amount: Amount to use for fake response\n        :return: Result from either cancel_order if usable, or fetch_order\n        "
        corder = self.cancel_stoploss_order(order_id, pair)
        if self.is_cancel_order_result_suitable(corder):
            return corder
        try:
            order = self.fetch_stoploss_order(order_id, pair)
        except InvalidOrderException:
            logger.warning(f'Could not fetch cancelled stoploss order {order_id}.')
            order = {'fee': {}, 'status': 'canceled', 'amount': amount, 'info': {}}
        return order

    @retrier
    def get_balances(self) -> dict:
        if False:
            print('Hello World!')
        try:
            balances = self._api.fetch_balance()
            balances.pop('info', None)
            balances.pop('free', None)
            balances.pop('total', None)
            balances.pop('used', None)
            return balances
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get balance due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def fetch_positions(self, pair: Optional[str]=None) -> List[Dict]:
        if False:
            while True:
                i = 10
        '\n        Fetch positions from the exchange.\n        If no pair is given, all positions are returned.\n        :param pair: Pair for the query\n        '
        if self._config['dry_run'] or self.trading_mode != TradingMode.FUTURES:
            return []
        try:
            symbols = []
            if pair:
                symbols.append(pair)
            positions: List[Dict] = self._api.fetch_positions(symbols)
            self._log_exchange_response('fetch_positions', positions)
            return positions
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get positions due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _fetch_orders_emulate(self, pair: str, since_ms: int) -> List[Dict]:
        if False:
            return 10
        orders = []
        if self.exchange_has('fetchClosedOrders'):
            orders = self._api.fetch_closed_orders(pair, since=since_ms)
            if self.exchange_has('fetchOpenOrders'):
                orders_open = self._api.fetch_open_orders(pair, since=since_ms)
                orders.extend(orders_open)
        return orders

    @retrier(retries=0)
    def fetch_orders(self, pair: str, since: datetime, params: Optional[Dict]=None) -> List[Dict]:
        if False:
            return 10
        '\n        Fetch all orders for a pair "since"\n        :param pair: Pair for the query\n        :param since: Starting time for the query\n        '
        if self._config['dry_run']:
            return []
        try:
            since_ms = int((since.timestamp() - 10) * 1000)
            if self.exchange_has('fetchOrders'):
                if not params:
                    params = {}
                try:
                    orders: List[Dict] = self._api.fetch_orders(pair, since=since_ms, params=params)
                except ccxt.NotSupported:
                    orders = self._fetch_orders_emulate(pair, since_ms)
            else:
                orders = self._fetch_orders_emulate(pair, since_ms)
            self._log_exchange_response('fetch_orders', orders)
            orders = [self._order_contracts_to_amount(o) for o in orders]
            return orders
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not fetch positions due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def fetch_trading_fees(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Fetch user account trading fees\n        Can be cached, should not update often.\n        '
        if self._config['dry_run'] or self.trading_mode != TradingMode.FUTURES or (not self.exchange_has('fetchTradingFees')):
            return {}
        try:
            trading_fees: Dict[str, Any] = self._api.fetch_trading_fees()
            self._log_exchange_response('fetch_trading_fees', trading_fees)
            return trading_fees
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not fetch trading fees due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def fetch_bids_asks(self, symbols: Optional[List[str]]=None, cached: bool=False) -> Dict:
        if False:
            return 10
        '\n        :param cached: Allow cached result\n        :return: fetch_tickers result\n        '
        if not self.exchange_has('fetchBidsAsks'):
            return {}
        if cached:
            with self._cache_lock:
                tickers = self._fetch_tickers_cache.get('fetch_bids_asks')
            if tickers:
                return tickers
        try:
            tickers = self._api.fetch_bids_asks(symbols)
            with self._cache_lock:
                self._fetch_tickers_cache['fetch_bids_asks'] = tickers
            return tickers
        except ccxt.NotSupported as e:
            raise OperationalException(f'Exchange {self._api.name} does not support fetching bids/asks in batch. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load bids/asks due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_tickers(self, symbols: Optional[List[str]]=None, cached: bool=False) -> Tickers:
        if False:
            print('Hello World!')
        '\n        :param cached: Allow cached result\n        :return: fetch_tickers result\n        '
        tickers: Tickers
        if not self.exchange_has('fetchTickers'):
            return {}
        if cached:
            with self._cache_lock:
                tickers = self._fetch_tickers_cache.get('fetch_tickers')
            if tickers:
                return tickers
        try:
            tickers = self._api.fetch_tickers(symbols)
            with self._cache_lock:
                self._fetch_tickers_cache['fetch_tickers'] = tickers
            return tickers
        except ccxt.NotSupported as e:
            raise OperationalException(f'Exchange {self._api.name} does not support fetching tickers in batch. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load tickers due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def fetch_ticker(self, pair: str) -> Ticker:
        if False:
            print('Hello World!')
        try:
            if pair not in self.markets or self.markets[pair].get('active', False) is False:
                raise ExchangeError(f'Pair {pair} not available')
            data: Ticker = self._api.fetch_ticker(pair)
            return data
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load ticker due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @staticmethod
    def get_next_limit_in_list(limit: int, limit_range: Optional[List[int]], range_required: bool=True):
        if False:
            while True:
                i = 10
        '\n        Get next greater value in the list.\n        Used by fetch_l2_order_book if the api only supports a limited range\n        '
        if not limit_range:
            return limit
        result = min([x for x in limit_range if limit <= x] + [max(limit_range)])
        if not range_required and limit > result:
            return None
        return result

    @retrier
    def fetch_l2_order_book(self, pair: str, limit: int=100) -> OrderBook:
        if False:
            i = 10
            return i + 15
        "\n        Get L2 order book from exchange.\n        Can be limited to a certain amount (if supported).\n        Returns a dict in the format\n        {'asks': [price, volume], 'bids': [price, volume]}\n        "
        limit1 = self.get_next_limit_in_list(limit, self._ft_has['l2_limit_range'], self._ft_has['l2_limit_range_required'])
        try:
            return self._api.fetch_l2_order_book(pair, limit1)
        except ccxt.NotSupported as e:
            raise OperationalException(f'Exchange {self._api.name} does not support fetching order book.Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get order book due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _get_price_side(self, side: str, is_short: bool, conf_strategy: Dict) -> BidAsk:
        if False:
            print('Hello World!')
        price_side = conf_strategy['price_side']
        if price_side in ('same', 'other'):
            price_map = {('entry', 'long', 'same'): 'bid', ('entry', 'long', 'other'): 'ask', ('entry', 'short', 'same'): 'ask', ('entry', 'short', 'other'): 'bid', ('exit', 'long', 'same'): 'ask', ('exit', 'long', 'other'): 'bid', ('exit', 'short', 'same'): 'bid', ('exit', 'short', 'other'): 'ask'}
            price_side = price_map[side, 'short' if is_short else 'long', price_side]
        return price_side

    def get_rate(self, pair: str, refresh: bool, side: EntryExit, is_short: bool, order_book: Optional[OrderBook]=None, ticker: Optional[Ticker]=None) -> float:
        if False:
            while True:
                i = 10
        '\n        Calculates bid/ask target\n        bid rate - between current ask price and last price\n        ask rate - either using ticker bid or first bid based on orderbook\n        or remain static in any other case since it\'s not updating.\n        :param pair: Pair to get rate for\n        :param refresh: allow cached data\n        :param side: "buy" or "sell"\n        :return: float: Price\n        :raises PricingError if orderbook price could not be determined.\n        '
        name = side.capitalize()
        strat_name = 'entry_pricing' if side == 'entry' else 'exit_pricing'
        cache_rate: TTLCache = self._entry_rate_cache if side == 'entry' else self._exit_rate_cache
        if not refresh:
            with self._cache_lock:
                rate = cache_rate.get(pair)
            if rate:
                logger.debug(f'Using cached {side} rate for {pair}.')
                return rate
        conf_strategy = self._config.get(strat_name, {})
        price_side = self._get_price_side(side, is_short, conf_strategy)
        if conf_strategy.get('use_order_book', False):
            order_book_top = conf_strategy.get('order_book_top', 1)
            if order_book is None:
                order_book = self.fetch_l2_order_book(pair, order_book_top)
            rate = self._get_rate_from_ob(pair, side, order_book, name, price_side, order_book_top)
        else:
            logger.debug(f'Using Last {price_side.capitalize()} / Last Price')
            if ticker is None:
                ticker = self.fetch_ticker(pair)
            rate = self._get_rate_from_ticker(side, ticker, conf_strategy, price_side)
        if rate is None:
            raise PricingError(f'{name}-Rate for {pair} was empty.')
        with self._cache_lock:
            cache_rate[pair] = rate
        return rate

    def _get_rate_from_ticker(self, side: EntryExit, ticker: Ticker, conf_strategy: Dict[str, Any], price_side: BidAsk) -> Optional[float]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get rate from ticker.\n        '
        ticker_rate = ticker[price_side]
        if ticker['last'] and ticker_rate:
            if side == 'entry' and ticker_rate > ticker['last']:
                balance = conf_strategy.get('price_last_balance', 0.0)
                ticker_rate = ticker_rate + balance * (ticker['last'] - ticker_rate)
            elif side == 'exit' and ticker_rate < ticker['last']:
                balance = conf_strategy.get('price_last_balance', 0.0)
                ticker_rate = ticker_rate - balance * (ticker_rate - ticker['last'])
        rate = ticker_rate
        return rate

    def _get_rate_from_ob(self, pair: str, side: EntryExit, order_book: OrderBook, name: str, price_side: BidAsk, order_book_top: int) -> float:
        if False:
            return 10
        '\n        Get rate from orderbook\n        :raises: PricingError if rate could not be determined.\n        '
        logger.debug('order_book %s', order_book)
        try:
            obside: OBLiteral = 'bids' if price_side == 'bid' else 'asks'
            rate = order_book[obside][order_book_top - 1][0]
        except (IndexError, KeyError) as e:
            logger.warning(f'{pair} - {name} Price at location {order_book_top} from orderbook could not be determined. Orderbook: {order_book}')
            raise PricingError from e
        logger.debug(f'{pair} - {name} price from orderbook {price_side.capitalize()}side - top {order_book_top} order book {side} rate {rate:.8f}')
        return rate

    def get_rates(self, pair: str, refresh: bool, is_short: bool) -> Tuple[float, float]:
        if False:
            return 10
        entry_rate = None
        exit_rate = None
        if not refresh:
            with self._cache_lock:
                entry_rate = self._entry_rate_cache.get(pair)
                exit_rate = self._exit_rate_cache.get(pair)
            if entry_rate:
                logger.debug(f'Using cached buy rate for {pair}.')
            if exit_rate:
                logger.debug(f'Using cached sell rate for {pair}.')
        entry_pricing = self._config.get('entry_pricing', {})
        exit_pricing = self._config.get('exit_pricing', {})
        order_book = ticker = None
        if not entry_rate and entry_pricing.get('use_order_book', False):
            order_book_top = max(entry_pricing.get('order_book_top', 1), exit_pricing.get('order_book_top', 1))
            order_book = self.fetch_l2_order_book(pair, order_book_top)
            entry_rate = self.get_rate(pair, refresh, 'entry', is_short, order_book=order_book)
        elif not entry_rate:
            ticker = self.fetch_ticker(pair)
            entry_rate = self.get_rate(pair, refresh, 'entry', is_short, ticker=ticker)
        if not exit_rate:
            exit_rate = self.get_rate(pair, refresh, 'exit', is_short, order_book=order_book, ticker=ticker)
        return (entry_rate, exit_rate)

    @retrier
    def get_trades_for_order(self, order_id: str, pair: str, since: datetime, params: Optional[Dict]=None) -> List:
        if False:
            while True:
                i = 10
        '\n        Fetch Orders using the "fetch_my_trades" endpoint and filter them by order-id.\n        The "since" argument passed in is coming from the database and is in UTC,\n        as timezone-native datetime object.\n        From the python documentation:\n            > Naive datetime instances are assumed to represent local time\n        Therefore, calling "since.timestamp()" will get the UTC timestamp, after applying the\n        transformation from local timezone to UTC.\n        This works for timezones UTC+ since then the result will contain trades from a few hours\n        instead of from the last 5 seconds, however fails for UTC- timezones,\n        since we\'re then asking for trades with a "since" argument in the future.\n\n        :param order_id order_id: Order-id as given when creating the order\n        :param pair: Pair the order is for\n        :param since: datetime object of the order creation time. Assumes object is in UTC.\n        '
        if self._config['dry_run']:
            return []
        if not self.exchange_has('fetchMyTrades'):
            return []
        try:
            _params = params if params else {}
            my_trades = self._api.fetch_my_trades(pair, int((since.replace(tzinfo=timezone.utc).timestamp() - 5) * 1000), params=_params)
            matched_trades = [trade for trade in my_trades if trade['order'] == order_id]
            self._log_exchange_response('get_trades_for_order', matched_trades)
            matched_trades = self._trades_contracts_to_amount(matched_trades)
            return matched_trades
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get trades due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_order_id_conditional(self, order: Dict[str, Any]) -> str:
        if False:
            return 10
        return order['id']

    @retrier
    def get_fee(self, symbol: str, type: str='', side: str='', amount: float=1, price: float=1, taker_or_maker: MakerTaker='maker') -> float:
        if False:
            while True:
                i = 10
        '\n        Retrieve fee from exchange\n        :param symbol: Pair\n        :param type: Type of order (market, limit, ...)\n        :param side: Side of order (buy, sell)\n        :param amount: Amount of order\n        :param price: Price of order\n        :param taker_or_maker: \'maker\' or \'taker\' (ignored if "type" is provided)\n        '
        if type and type == 'market':
            taker_or_maker = 'taker'
        try:
            if self._config['dry_run'] and self._config.get('fee', None) is not None:
                return self._config['fee']
            if self._api.markets is None or len(self._api.markets) == 0:
                self._api.load_markets(params={})
            return self._api.calculate_fee(symbol=symbol, type=type, side=side, amount=amount, price=price, takerOrMaker=taker_or_maker)['rate']
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get fee info due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @staticmethod
    def order_has_fee(order: Dict) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Verifies if the passed in order dict has the needed keys to extract fees,\n        and that these keys (currency, cost) are not empty.\n        :param order: Order or trade (one trade) dict\n        :return: True if the fee substructure contains currency and cost, false otherwise\n        '
        if not isinstance(order, dict):
            return False
        return 'fee' in order and order['fee'] is not None and (order['fee'].keys() >= {'currency', 'cost'}) and (order['fee']['currency'] is not None) and (order['fee']['cost'] is not None)

    def calculate_fee_rate(self, fee: Dict, symbol: str, cost: float, amount: float) -> Optional[float]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Calculate fee rate if it's not given by the exchange.\n        :param fee: ccxt Fee dict - must contain cost / currency / rate\n        :param symbol: Symbol of the order\n        :param cost: Total cost of the order\n        :param amount: Amount of the order\n        "
        if fee.get('rate') is not None:
            return fee.get('rate')
        fee_curr = fee.get('currency')
        if fee_curr is None:
            return None
        fee_cost = float(fee['cost'])
        if fee_curr == self.get_pair_base_currency(symbol):
            return round(fee_cost / amount, 8)
        elif fee_curr == self.get_pair_quote_currency(symbol):
            return round(fee_cost / cost, 8) if cost else None
        else:
            if not cost:
                return None
            try:
                comb = self.get_valid_pair_combination(fee_curr, self._config['stake_currency'])
                tick = self.fetch_ticker(comb)
                fee_to_quote_rate = safe_value_fallback2(tick, tick, 'last', 'ask')
            except (ValueError, ExchangeError):
                fee_to_quote_rate = self._config['exchange'].get('unknown_fee_rate', None)
                if not fee_to_quote_rate:
                    return None
            return round(fee_cost * fee_to_quote_rate / cost, 8)

    def extract_cost_curr_rate(self, fee: Dict, symbol: str, cost: float, amount: float) -> Tuple[float, str, Optional[float]]:
        if False:
            while True:
                i = 10
        '\n        Extract tuple of cost, currency, rate.\n        Requires order_has_fee to run first!\n        :param fee: ccxt Fee dict - must contain cost / currency / rate\n        :param symbol: Symbol of the order\n        :param cost: Total cost of the order\n        :param amount: Amount of the order\n        :return: Tuple with cost, currency, rate of the given fee dict\n        '
        return (float(fee['cost']), fee['currency'], self.calculate_fee_rate(fee, symbol, cost, amount))

    def get_historic_ohlcv(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType, is_new_pair: bool=False, until_ms: Optional[int]=None) -> List:
        if False:
            for i in range(10):
                print('nop')
        "\n        Get candle history using asyncio and returns the list of candles.\n        Handles all async work for this.\n        Async over one pair, assuming we get `self.ohlcv_candle_limit()` candles per call.\n        :param pair: Pair to download\n        :param timeframe: Timeframe to get data for\n        :param since_ms: Timestamp in milliseconds to get history from\n        :param until_ms: Timestamp in milliseconds to get history up to\n        :param candle_type: '', mark, index, premiumIndex, or funding_rate\n        :return: List with candle (OHLCV) data\n        "
        (pair, _, _, data, _) = self.loop.run_until_complete(self._async_get_historic_ohlcv(pair=pair, timeframe=timeframe, since_ms=since_ms, until_ms=until_ms, is_new_pair=is_new_pair, candle_type=candle_type))
        logger.info(f'Downloaded data for {pair} with length {len(data)}.')
        return data

    async def _async_get_historic_ohlcv(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType, is_new_pair: bool=False, raise_: bool=False, until_ms: Optional[int]=None) -> OHLCVResponse:
        """
        Download historic ohlcv
        :param is_new_pair: used by binance subclass to allow "fast" new pair downloading
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """
        one_call = timeframe_to_msecs(timeframe) * self.ohlcv_candle_limit(timeframe, candle_type, since_ms)
        logger.debug('one_call: %s msecs (%s)', one_call, dt_humanize(dt_now() - timedelta(milliseconds=one_call), only_distance=True))
        input_coroutines = [self._async_get_candle_history(pair, timeframe, candle_type, since) for since in range(since_ms, until_ms or dt_ts(), one_call)]
        data: List = []
        for input_coro in chunks(input_coroutines, 100):
            results = await asyncio.gather(*input_coro, return_exceptions=True)
            for res in results:
                if isinstance(res, BaseException):
                    logger.warning(f'Async code raised an exception: {repr(res)}')
                    if raise_:
                        raise
                    continue
                else:
                    (p, _, c, new_data, _) = res
                    if p == pair and c == candle_type:
                        data.extend(new_data)
        data = sorted(data, key=lambda x: x[0])
        return (pair, timeframe, candle_type, data, self._ohlcv_partial_candle)

    def _build_coroutine(self, pair: str, timeframe: str, candle_type: CandleType, since_ms: Optional[int], cache: bool) -> Coroutine[Any, Any, OHLCVResponse]:
        if False:
            i = 10
            return i + 15
        not_all_data = cache and self.required_candle_call_count > 1
        if cache and (pair, timeframe, candle_type) in self._klines:
            candle_limit = self.ohlcv_candle_limit(timeframe, candle_type)
            min_date = date_minus_candles(timeframe, candle_limit - 5).timestamp()
            if min_date < self._pairs_last_refresh_time.get((pair, timeframe, candle_type), 0):
                not_all_data = False
            else:
                logger.info(f'Time jump detected. Evicting cache for {pair}, {timeframe}, {candle_type}')
                del self._klines[pair, timeframe, candle_type]
        if not since_ms and (self._ft_has['ohlcv_require_since'] or not_all_data):
            one_call = timeframe_to_msecs(timeframe) * self.ohlcv_candle_limit(timeframe, candle_type, since_ms)
            move_to = one_call * self.required_candle_call_count
            now = timeframe_to_next_date(timeframe)
            since_ms = int((now - timedelta(seconds=move_to // 1000)).timestamp() * 1000)
        if since_ms:
            return self._async_get_historic_ohlcv(pair, timeframe, since_ms=since_ms, raise_=True, candle_type=candle_type)
        else:
            return self._async_get_candle_history(pair, timeframe, since_ms=since_ms, candle_type=candle_type)

    def _build_ohlcv_dl_jobs(self, pair_list: ListPairsWithTimeframes, since_ms: Optional[int], cache: bool) -> Tuple[List[Coroutine], List[Tuple[str, str, CandleType]]]:
        if False:
            print('Hello World!')
        '\n        Build Coroutines to execute as part of refresh_latest_ohlcv\n        '
        input_coroutines: List[Coroutine[Any, Any, OHLCVResponse]] = []
        cached_pairs = []
        for (pair, timeframe, candle_type) in set(pair_list):
            if timeframe not in self.timeframes and candle_type in (CandleType.SPOT, CandleType.FUTURES):
                logger.warning(f"Cannot download ({pair}, {timeframe}) combination as this timeframe is not available on {self.name}. Available timeframes are {', '.join(self.timeframes)}.")
                continue
            if (pair, timeframe, candle_type) not in self._klines or not cache or self._now_is_time_to_refresh(pair, timeframe, candle_type):
                input_coroutines.append(self._build_coroutine(pair, timeframe, candle_type, since_ms, cache))
            else:
                logger.debug(f'Using cached candle (OHLCV) data for {pair}, {timeframe}, {candle_type} ...')
                cached_pairs.append((pair, timeframe, candle_type))
        return (input_coroutines, cached_pairs)

    def _process_ohlcv_df(self, pair: str, timeframe: str, c_type: CandleType, ticks: List[List], cache: bool, drop_incomplete: bool) -> DataFrame:
        if False:
            while True:
                i = 10
        if ticks and cache:
            idx = -2 if drop_incomplete and len(ticks) > 1 else -1
            self._pairs_last_refresh_time[pair, timeframe, c_type] = ticks[idx][0] // 1000
        ohlcv_df = ohlcv_to_dataframe(ticks, timeframe, pair=pair, fill_missing=True, drop_incomplete=drop_incomplete)
        if cache:
            if (pair, timeframe, c_type) in self._klines:
                old = self._klines[pair, timeframe, c_type]
                ohlcv_df = clean_ohlcv_dataframe(concat([old, ohlcv_df], axis=0), timeframe, pair, fill_missing=True, drop_incomplete=False)
                candle_limit = self.ohlcv_candle_limit(timeframe, self._config['candle_type_def'])
                ohlcv_df = ohlcv_df.tail(candle_limit + self._startup_candle_count)
                ohlcv_df = ohlcv_df.reset_index(drop=True)
                self._klines[pair, timeframe, c_type] = ohlcv_df
            else:
                self._klines[pair, timeframe, c_type] = ohlcv_df
        return ohlcv_df

    def refresh_latest_ohlcv(self, pair_list: ListPairsWithTimeframes, *, since_ms: Optional[int]=None, cache: bool=True, drop_incomplete: Optional[bool]=None) -> Dict[PairWithTimeframe, DataFrame]:
        if False:
            print('Hello World!')
        '\n        Refresh in-memory OHLCV asynchronously and set `_klines` with the result\n        Loops asynchronously over pair_list and downloads all pairs async (semi-parallel).\n        Only used in the dataprovider.refresh() method.\n        :param pair_list: List of 2 element tuples containing pair, interval to refresh\n        :param since_ms: time since when to download, in milliseconds\n        :param cache: Assign result to _klines. Usefull for one-off downloads like for pairlists\n        :param drop_incomplete: Control candle dropping.\n            Specifying None defaults to _ohlcv_partial_candle\n        :return: Dict of [{(pair, timeframe): Dataframe}]\n        '
        logger.debug('Refreshing candle (OHLCV) data for %d pairs', len(pair_list))
        (input_coroutines, cached_pairs) = self._build_ohlcv_dl_jobs(pair_list, since_ms, cache)
        results_df = {}
        for input_coro in chunks(input_coroutines, 100):

            async def gather_stuff():
                return await asyncio.gather(*input_coro, return_exceptions=True)
            with self._loop_lock:
                results = self.loop.run_until_complete(gather_stuff())
            for res in results:
                if isinstance(res, Exception):
                    logger.warning(f'Async code raised an exception: {repr(res)}')
                    continue
                (pair, timeframe, c_type, ticks, drop_hint) = res
                drop_incomplete_ = drop_hint if drop_incomplete is None else drop_incomplete
                ohlcv_df = self._process_ohlcv_df(pair, timeframe, c_type, ticks, cache, drop_incomplete_)
                results_df[pair, timeframe, c_type] = ohlcv_df
        for (pair, timeframe, c_type) in cached_pairs:
            results_df[pair, timeframe, c_type] = self.klines((pair, timeframe, c_type), copy=False)
        return results_df

    def _now_is_time_to_refresh(self, pair: str, timeframe: str, candle_type: CandleType) -> bool:
        if False:
            while True:
                i = 10
        interval_in_sec = timeframe_to_seconds(timeframe)
        plr = self._pairs_last_refresh_time.get((pair, timeframe, candle_type), 0) + interval_in_sec
        now = int(timeframe_to_prev_date(timeframe).timestamp())
        return plr < now

    @retrier_async
    async def _async_get_candle_history(self, pair: str, timeframe: str, candle_type: CandleType, since_ms: Optional[int]=None) -> OHLCVResponse:
        """
        Asynchronously get candle history data using fetch_ohlcv
        :param candle_type: '', mark, index, premiumIndex, or funding_rate
        returns tuple: (pair, timeframe, ohlcv_list)
        """
        try:
            s = '(' + dt_from_ts(since_ms).isoformat() + ') ' if since_ms is not None else ''
            logger.debug('Fetching pair %s, %s, interval %s, since %s %s...', pair, candle_type, timeframe, since_ms, s)
            params = deepcopy(self._ft_has.get('ohlcv_params', {}))
            candle_limit = self.ohlcv_candle_limit(timeframe, candle_type=candle_type, since_ms=since_ms)
            if candle_type and candle_type != CandleType.SPOT:
                params.update({'price': candle_type.value})
            if candle_type != CandleType.FUNDING_RATE:
                data = await self._api_async.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms, limit=candle_limit, params=params)
            else:
                data = await self._fetch_funding_rate_history(pair=pair, timeframe=timeframe, limit=candle_limit, since_ms=since_ms)
            try:
                if data and data[0][0] > data[-1][0]:
                    data = sorted(data, key=lambda x: x[0])
            except IndexError:
                logger.exception('Error loading %s. Result was %s.', pair, data)
                return (pair, timeframe, candle_type, [], self._ohlcv_partial_candle)
            logger.debug('Done fetching pair %s, %s interval %s...', pair, candle_type, timeframe)
            return (pair, timeframe, candle_type, data, self._ohlcv_partial_candle)
        except ccxt.NotSupported as e:
            raise OperationalException(f'Exchange {self._api.name} does not support fetching historical candle (OHLCV) data. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not fetch historical candle (OHLCV) data for pair {pair} due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch historical candle (OHLCV) data for pair {pair}. Message: {e}') from e

    async def _fetch_funding_rate_history(self, pair: str, timeframe: str, limit: int, since_ms: Optional[int]=None) -> List[List]:
        """
        Fetch funding rate history - used to selectively override this by subclasses.
        """
        data = await self._api_async.fetch_funding_rate_history(pair, since=since_ms, limit=limit)
        data = [[x['timestamp'], x['fundingRate'], 0, 0, 0, 0] for x in data]
        return data

    @retrier_async
    async def _async_fetch_trades(self, pair: str, since: Optional[int]=None, params: Optional[dict]=None) -> List[List]:
        """
        Asyncronously gets trade history using fetch_trades.
        Handles exchange errors, does one call to the exchange.
        :param pair: Pair to fetch trade data for
        :param since: Since as integer timestamp in milliseconds
        returns: List of dicts containing trades
        """
        try:
            if params:
                logger.debug('Fetching trades for pair %s, params: %s ', pair, params)
                trades = await self._api_async.fetch_trades(pair, params=params, limit=1000)
            else:
                logger.debug('Fetching trades for pair %s, since %s %s...', pair, since, '(' + dt_from_ts(since).isoformat() + ') ' if since is not None else '')
                trades = await self._api_async.fetch_trades(pair, since=since, limit=1000)
            trades = self._trades_contracts_to_amount(trades)
            return trades_dict_to_list(trades)
        except ccxt.NotSupported as e:
            raise OperationalException(f'Exchange {self._api.name} does not support fetching historical trade data.Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load trade history due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch trade data. Msg: {e}') from e

    async def _async_get_trade_history_id(self, pair: str, until: int, since: Optional[int]=None, from_id: Optional[str]=None) -> Tuple[str, List[List]]:
        """
        Asyncronously gets trade history using fetch_trades
        use this when exchange uses id-based iteration (check `self._trades_pagination`)
        :param pair: Pair to fetch trade data for
        :param since: Since as integer timestamp in milliseconds
        :param until: Until as integer timestamp in milliseconds
        :param from_id: Download data starting with ID (if id is known). Ignores "since" if set.
        returns tuple: (pair, trades-list)
        """
        trades: List[List] = []
        if not from_id:
            t = await self._async_fetch_trades(pair, since=since)
            from_id = t[-1][1]
            trades.extend(t[:-1])
        while True:
            try:
                t = await self._async_fetch_trades(pair, params={self._trades_pagination_arg: from_id})
                if t:
                    trades.extend(t[:-1])
                    if from_id == t[-1][1] or t[-1][0] > until:
                        logger.debug(f'Stopping because from_id did not change. Reached {t[-1][0]} > {until}')
                        trades.extend(t[-1:])
                        break
                    from_id = t[-1][1]
                else:
                    logger.debug('Stopping as no more trades were returned.')
                    break
            except asyncio.CancelledError:
                logger.debug('Async operation Interrupted, breaking trades DL loop.')
                break
        return (pair, trades)

    async def _async_get_trade_history_time(self, pair: str, until: int, since: Optional[int]=None) -> Tuple[str, List[List]]:
        """
        Asyncronously gets trade history using fetch_trades,
        when the exchange uses time-based iteration (check `self._trades_pagination`)
        :param pair: Pair to fetch trade data for
        :param since: Since as integer timestamp in milliseconds
        :param until: Until as integer timestamp in milliseconds
        returns tuple: (pair, trades-list)
        """
        trades: List[List] = []
        while True:
            try:
                t = await self._async_fetch_trades(pair, since=since)
                if t:
                    if since == t[-1][0] and len(t) == 1:
                        logger.debug('Stopping because no more trades are available.')
                        break
                    since = t[-1][0]
                    trades.extend(t)
                    if until and t[-1][0] > until:
                        logger.debug(f'Stopping because until was reached. {t[-1][0]} > {until}')
                        break
                else:
                    logger.debug('Stopping as no more trades were returned.')
                    break
            except asyncio.CancelledError:
                logger.debug('Async operation Interrupted, breaking trades DL loop.')
                break
        return (pair, trades)

    async def _async_get_trade_history(self, pair: str, since: Optional[int]=None, until: Optional[int]=None, from_id: Optional[str]=None) -> Tuple[str, List[List]]:
        """
        Async wrapper handling downloading trades using either time or id based methods.
        """
        logger.debug(f'_async_get_trade_history(), pair: {pair}, since: {since}, until: {until}, from_id: {from_id}')
        if until is None:
            until = ccxt.Exchange.milliseconds()
            logger.debug(f'Exchange milliseconds: {until}')
        if self._trades_pagination == 'time':
            return await self._async_get_trade_history_time(pair=pair, since=since, until=until)
        elif self._trades_pagination == 'id':
            return await self._async_get_trade_history_id(pair=pair, since=since, until=until, from_id=from_id)
        else:
            raise OperationalException(f'Exchange {self.name} does use neither time, nor id based pagination')

    def get_historic_trades(self, pair: str, since: Optional[int]=None, until: Optional[int]=None, from_id: Optional[str]=None) -> Tuple[str, List]:
        if False:
            return 10
        '\n        Get trade history data using asyncio.\n        Handles all async work and returns the list of candles.\n        Async over one pair, assuming we get `self.ohlcv_candle_limit()` candles per call.\n        :param pair: Pair to download\n        :param since: Timestamp in milliseconds to get history from\n        :param until: Timestamp in milliseconds. Defaults to current timestamp if not defined.\n        :param from_id: Download data starting with ID (if id is known)\n        :returns List of trade data\n        '
        if not self.exchange_has('fetchTrades'):
            raise OperationalException('This exchange does not support downloading Trades.')
        with self._loop_lock:
            task = asyncio.ensure_future(self._async_get_trade_history(pair=pair, since=since, until=until, from_id=from_id))
            for sig in [signal.SIGINT, signal.SIGTERM]:
                try:
                    self.loop.add_signal_handler(sig, task.cancel)
                except NotImplementedError:
                    pass
            return self.loop.run_until_complete(task)

    @retrier
    def _get_funding_fees_from_exchange(self, pair: str, since: Union[datetime, int]) -> float:
        if False:
            while True:
                i = 10
        '\n        Returns the sum of all funding fees that were exchanged for a pair within a timeframe\n        Dry-run handling happens as part of _calculate_funding_fees.\n        :param pair: (e.g. ADA/USDT)\n        :param since: The earliest time of consideration for calculating funding fees,\n            in unix time or as a datetime\n        '
        if not self.exchange_has('fetchFundingHistory'):
            raise OperationalException(f'fetch_funding_history() is not available using {self.name}')
        if type(since) is datetime:
            since = int(since.timestamp()) * 1000
        try:
            funding_history = self._api.fetch_funding_history(symbol=pair, since=since)
            return sum((fee['amount'] for fee in funding_history))
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get funding fees due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_leverage_tiers(self) -> Dict[str, List[Dict]]:
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._api.fetch_leverage_tiers()
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load leverage tiers due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier_async
    async def get_market_leverage_tiers(self, symbol: str) -> Tuple[str, List[Dict]]:
        """ Leverage tiers per symbol """
        try:
            tier = await self._api_async.fetch_market_leverage_tiers(symbol)
            return (symbol, tier)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load leverage tiers for {symbol} due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def load_leverage_tiers(self) -> Dict[str, List[Dict]]:
        if False:
            while True:
                i = 10
        if self.trading_mode == TradingMode.FUTURES:
            if self.exchange_has('fetchLeverageTiers'):
                return self.get_leverage_tiers()
            elif self.exchange_has('fetchMarketLeverageTiers'):
                markets = self.markets
                symbols = [symbol for (symbol, market) in markets.items() if self.market_is_future(market) and market['quote'] == self._config['stake_currency']]
                tiers: Dict[str, List[Dict]] = {}
                tiers_cached = self.load_cached_leverage_tiers(self._config['stake_currency'])
                if tiers_cached:
                    tiers = tiers_cached
                coros = [self.get_market_leverage_tiers(symbol) for symbol in sorted(symbols) if symbol not in tiers]
                if coros:
                    logger.info(f'Initializing leverage_tiers for {len(symbols)} markets. This will take about a minute.')
                else:
                    logger.info('Using cached leverage_tiers.')

                async def gather_results(input_coro):
                    return await asyncio.gather(*input_coro, return_exceptions=True)
                for input_coro in chunks(coros, 100):
                    with self._loop_lock:
                        results = self.loop.run_until_complete(gather_results(input_coro))
                    for res in results:
                        if isinstance(res, Exception):
                            logger.warning(f'Leverage tier exception: {repr(res)}')
                            continue
                        (symbol, tier) = res
                        tiers[symbol] = tier
                if len(coros) > 0:
                    self.cache_leverage_tiers(tiers, self._config['stake_currency'])
                logger.info(f'Done initializing {len(symbols)} markets.')
                return tiers
        return {}

    def cache_leverage_tiers(self, tiers: Dict[str, List[Dict]], stake_currency: str) -> None:
        if False:
            return 10
        filename = self._config['datadir'] / 'futures' / f'leverage_tiers_{stake_currency}.json'
        if not filename.parent.is_dir():
            filename.parent.mkdir(parents=True)
        data = {'updated': datetime.now(timezone.utc), 'data': tiers}
        file_dump_json(filename, data)

    def load_cached_leverage_tiers(self, stake_currency: str) -> Optional[Dict[str, List[Dict]]]:
        if False:
            i = 10
            return i + 15
        filename = self._config['datadir'] / 'futures' / f'leverage_tiers_{stake_currency}.json'
        if filename.is_file():
            try:
                tiers = file_load_json(filename)
                updated = tiers.get('updated')
                if updated:
                    updated_dt = parser.parse(updated)
                    if updated_dt < datetime.now(timezone.utc) - timedelta(weeks=4):
                        logger.info('Cached leverage tiers are outdated. Will update.')
                        return None
                return tiers['data']
            except Exception:
                logger.exception('Error loading cached leverage tiers. Refreshing.')
        return None

    def fill_leverage_tiers(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Assigns property _leverage_tiers to a dictionary of information about the leverage\n        allowed on each pair\n        '
        leverage_tiers = self.load_leverage_tiers()
        for (pair, tiers) in leverage_tiers.items():
            pair_tiers = []
            for tier in tiers:
                pair_tiers.append(self.parse_leverage_tier(tier))
            self._leverage_tiers[pair] = pair_tiers

    def parse_leverage_tier(self, tier) -> Dict:
        if False:
            print('Hello World!')
        info = tier.get('info', {})
        return {'minNotional': tier['minNotional'], 'maxNotional': tier['maxNotional'], 'maintenanceMarginRate': tier['maintenanceMarginRate'], 'maxLeverage': tier['maxLeverage'], 'maintAmt': float(info['cum']) if 'cum' in info else None}

    def get_max_leverage(self, pair: str, stake_amount: Optional[float]) -> float:
        if False:
            i = 10
            return i + 15
        '\n        Returns the maximum leverage that a pair can be traded at\n        :param pair: The base/quote currency pair being traded\n        :stake_amount: The total value of the traders margin_mode in quote currency\n        '
        if self.trading_mode == TradingMode.SPOT:
            return 1.0
        if self.trading_mode == TradingMode.FUTURES:
            if stake_amount is None:
                raise OperationalException(f'{self.name}.get_max_leverage requires argument stake_amount')
            if pair not in self._leverage_tiers:
                return 1.0
            pair_tiers = self._leverage_tiers[pair]
            if stake_amount == 0:
                return self._leverage_tiers[pair][0]['maxLeverage']
            for tier_index in range(len(pair_tiers)):
                tier = pair_tiers[tier_index]
                lev = tier['maxLeverage']
                if tier_index < len(pair_tiers) - 1:
                    next_tier = pair_tiers[tier_index + 1]
                    next_floor = next_tier['minNotional'] / next_tier['maxLeverage']
                    if next_floor > stake_amount:
                        return min(tier['maxNotional'] / stake_amount, lev)
                elif stake_amount > tier['maxNotional']:
                    raise InvalidOrderException(f'Amount {stake_amount} too high for {pair}')
                else:
                    return tier['maxLeverage']
            raise OperationalException('Looped through all tiers without finding a max leverage. Should never be reached')
        elif self.trading_mode == TradingMode.MARGIN:
            market = self.markets[pair]
            if market['limits']['leverage']['max'] is not None:
                return market['limits']['leverage']['max']
            else:
                return 1.0
        else:
            return 1.0

    @retrier
    def _set_leverage(self, leverage: float, pair: Optional[str]=None, accept_fail: bool=False):
        if False:
            return 10
        "\n        Set's the leverage before making a trade, in order to not\n        have the same leverage on every trade\n        "
        if self._config['dry_run'] or not self.exchange_has('setLeverage'):
            return
        if self._ft_has.get('floor_leverage', False) is True:
            leverage = floor(leverage)
        try:
            res = self._api.set_leverage(symbol=pair, leverage=leverage)
            self._log_exchange_response('set_leverage', res)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.BadRequest, ccxt.InsufficientFunds) as e:
            if not accept_fail:
                raise TemporaryError(f'Could not set leverage due to {e.__class__.__name__}. Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not set leverage due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_interest_rate(self) -> float:
        if False:
            return 10
        '\n        Retrieve interest rate - necessary for Margin trading.\n        Should not call the exchange directly when used from backtesting.\n        '
        return 0.0

    def funding_fee_cutoff(self, open_date: datetime) -> bool:
        if False:
            return 10
        '\n        Funding fees are only charged at full hours (usually every 4-8h).\n        Therefore a trade opening at 10:00:01 will not be charged a funding fee until the next hour.\n        :param open_date: The open date for a trade\n        :return: True if the date falls on a full hour, False otherwise\n        '
        return open_date.minute == 0 and open_date.second == 0

    @retrier
    def set_margin_mode(self, pair: str, margin_mode: MarginMode, accept_fail: bool=False, params: dict={}):
        if False:
            i = 10
            return i + 15
        '\n        Set\'s the margin mode on the exchange to cross or isolated for a specific pair\n        :param pair: base/quote currency pair (e.g. "ADA/USDT")\n        '
        if self._config['dry_run'] or not self.exchange_has('setMarginMode'):
            return
        try:
            res = self._api.set_margin_mode(margin_mode.value, pair, params)
            self._log_exchange_response('set_margin_mode', res)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except ccxt.BadRequest as e:
            if not accept_fail:
                raise TemporaryError(f'Could not set margin mode due to {e.__class__.__name__}. Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not set margin mode due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _fetch_and_calculate_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime, close_date: Optional[datetime]=None) -> float:
        if False:
            i = 10
            return i + 15
        '\n        Fetches and calculates the sum of all funding fees that occurred for a pair\n        during a futures trade.\n        Only used during dry-run or if the exchange does not provide a funding_rates endpoint.\n        :param pair: The quote/base pair of the trade\n        :param amount: The quantity of the trade\n        :param is_short: trade direction\n        :param open_date: The date and time that the trade started\n        :param close_date: The date and time that the trade ended\n        '
        if self.funding_fee_cutoff(open_date):
            open_date = timeframe_to_prev_date('1h', open_date)
        timeframe = self._ft_has['mark_ohlcv_timeframe']
        timeframe_ff = self._ft_has.get('funding_fee_timeframe', self._ft_has['mark_ohlcv_timeframe'])
        if not close_date:
            close_date = datetime.now(timezone.utc)
        since_ms = int(timeframe_to_prev_date(timeframe, open_date).timestamp()) * 1000
        mark_comb: PairWithTimeframe = (pair, timeframe, CandleType.from_string(self._ft_has['mark_ohlcv_price']))
        funding_comb: PairWithTimeframe = (pair, timeframe_ff, CandleType.FUNDING_RATE)
        candle_histories = self.refresh_latest_ohlcv([mark_comb, funding_comb], since_ms=since_ms, cache=False, drop_incomplete=False)
        try:
            funding_rates = candle_histories[funding_comb]
            mark_rates = candle_histories[mark_comb]
        except KeyError:
            raise ExchangeError('Could not find funding rates.') from None
        funding_mark_rates = self.combine_funding_and_mark(funding_rates, mark_rates)
        return self.calculate_funding_fees(funding_mark_rates, amount=amount, is_short=is_short, open_date=open_date, close_date=close_date)

    @staticmethod
    def combine_funding_and_mark(funding_rates: DataFrame, mark_rates: DataFrame, futures_funding_rate: Optional[int]=None) -> DataFrame:
        if False:
            return 10
        '\n        Combine funding-rates and mark-rates dataframes\n        :param funding_rates: Dataframe containing Funding rates (Type FUNDING_RATE)\n        :param mark_rates: Dataframe containing Mark rates (Type mark_ohlcv_price)\n        :param futures_funding_rate: Fake funding rate to use if funding_rates are not available\n        '
        if futures_funding_rate is None:
            return mark_rates.merge(funding_rates, on='date', how='inner', suffixes=['_mark', '_fund'])
        elif len(funding_rates) == 0:
            mark_rates['open_fund'] = futures_funding_rate
            return mark_rates.rename(columns={'open': 'open_mark', 'close': 'close_mark', 'high': 'high_mark', 'low': 'low_mark', 'volume': 'volume_mark'})
        else:
            combined = mark_rates.merge(funding_rates, on='date', how='outer', suffixes=['_mark', '_fund'])
            combined['open_fund'] = combined['open_fund'].fillna(futures_funding_rate)
            return combined

    def calculate_funding_fees(self, df: DataFrame, amount: float, is_short: bool, open_date: datetime, close_date: datetime, time_in_ratio: Optional[float]=None) -> float:
        if False:
            print('Hello World!')
        '\n        calculates the sum of all funding fees that occurred for a pair during a futures trade\n        :param df: Dataframe containing combined funding and mark rates\n                   as `open_fund` and `open_mark`.\n        :param amount: The quantity of the trade\n        :param is_short: trade direction\n        :param open_date: The date and time that the trade started\n        :param close_date: The date and time that the trade ended\n        :param time_in_ratio: Not used by most exchange classes\n        '
        fees: float = 0
        if not df.empty:
            df1 = df[(df['date'] >= open_date) & (df['date'] <= close_date)]
            fees = sum(df1['open_fund'] * df1['open_mark'] * amount)
        return fees if is_short else -fees

    def get_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime) -> float:
        if False:
            i = 10
            return i + 15
        '\n        Fetch funding fees, either from the exchange (live) or calculates them\n        based on funding rate/mark price history\n        :param pair: The quote/base pair of the trade\n        :param is_short: trade direction\n        :param amount: Trade amount\n        :param open_date: Open date of the trade\n        :return: funding fee since open_date\n        '
        if self.trading_mode == TradingMode.FUTURES:
            try:
                if self._config['dry_run']:
                    funding_fees = self._fetch_and_calculate_funding_fees(pair, amount, is_short, open_date)
                else:
                    funding_fees = self._get_funding_fees_from_exchange(pair, open_date)
                return funding_fees
            except ExchangeError:
                logger.warning(f'Could not update funding fees for {pair}.')
        return 0.0

    def get_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, mm_ex_1: float=0.0, upnl_ex_1: float=0.0) -> Optional[float]:
        if False:
            while True:
                i = 10
        "\n        Set's the margin mode on the exchange to cross or isolated for a specific pair\n        "
        if self.trading_mode == TradingMode.SPOT:
            return None
        elif self.trading_mode != TradingMode.FUTURES:
            raise OperationalException(f'{self.name} does not support {self.margin_mode} {self.trading_mode}')
        liquidation_price = None
        if self._config['dry_run'] or not self.exchange_has('fetchPositions'):
            liquidation_price = self.dry_run_liquidation_price(pair=pair, open_rate=open_rate, is_short=is_short, amount=amount, leverage=leverage, stake_amount=stake_amount, wallet_balance=wallet_balance, mm_ex_1=mm_ex_1, upnl_ex_1=upnl_ex_1)
        else:
            positions = self.fetch_positions(pair)
            if len(positions) > 0:
                pos = positions[0]
                liquidation_price = pos['liquidationPrice']
        if liquidation_price is not None:
            buffer_amount = abs(open_rate - liquidation_price) * self.liquidation_buffer
            liquidation_price_buffer = liquidation_price - buffer_amount if is_short else liquidation_price + buffer_amount
            return max(liquidation_price_buffer, 0.0)
        else:
            return None

    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, mm_ex_1: float=0.0, upnl_ex_1: float=0.0) -> Optional[float]:
        if False:
            return 10
        '\n        Important: Must be fetching data from cached values as this is used by backtesting!\n        PERPETUAL:\n         gate: https://www.gate.io/help/futures/futures/27724/liquidation-price-bankruptcy-price\n         > Liquidation Price = (Entry Price ± Margin / Contract Multiplier / Size) /\n                                [ 1 ± (Maintenance Margin Ratio + Taker Rate)]\n            Wherein, "+" or "-" depends on whether the contract goes long or short:\n            "-" for long, and "+" for short.\n\n         okex: https://www.okex.com/support/hc/en-us/articles/\n            360053909592-VI-Introduction-to-the-isolated-mode-of-Single-Multi-currency-Portfolio-margin\n\n        :param pair: Pair to calculate liquidation price for\n        :param open_rate: Entry price of position\n        :param is_short: True if the trade is a short, false otherwise\n        :param amount: Absolute value of position size incl. leverage (in base currency)\n        :param stake_amount: Stake amount - Collateral in settle currency.\n        :param leverage: Leverage used for this position.\n        :param trading_mode: SPOT, MARGIN, FUTURES, etc.\n        :param margin_mode: Either ISOLATED or CROSS\n        :param wallet_balance: Amount of margin_mode in the wallet being used to trade\n            Cross-Margin Mode: crossWalletBalance\n            Isolated-Margin Mode: isolatedWalletBalance\n\n        # * Not required by Gate or OKX\n        :param mm_ex_1:\n        :param upnl_ex_1:\n        '
        market = self.markets[pair]
        taker_fee_rate = market['taker']
        (mm_ratio, _) = self.get_maintenance_ratio_and_amt(pair, stake_amount)
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            if market['inverse']:
                raise OperationalException('Freqtrade does not yet support inverse contracts')
            value = wallet_balance / amount
            mm_ratio_taker = mm_ratio + taker_fee_rate
            if is_short:
                return (open_rate + value) / (1 + mm_ratio_taker)
            else:
                return (open_rate - value) / (1 - mm_ratio_taker)
        else:
            raise OperationalException('Freqtrade only supports isolated futures for leverage trading')

    def get_maintenance_ratio_and_amt(self, pair: str, nominal_value: float) -> Tuple[float, Optional[float]]:
        if False:
            return 10
        '\n        Important: Must be fetching data from cached values as this is used by backtesting!\n        :param pair: Market symbol\n        :param nominal_value: The total trade amount in quote currency including leverage\n        maintenance amount only on Binance\n        :return: (maintenance margin ratio, maintenance amount)\n        '
        if self._config.get('runmode') in OPTIMIZE_MODES or self.exchange_has('fetchLeverageTiers') or self.exchange_has('fetchMarketLeverageTiers'):
            if pair not in self._leverage_tiers:
                raise InvalidOrderException(f'Maintenance margin rate for {pair} is unavailable for {self.name}')
            pair_tiers = self._leverage_tiers[pair]
            for tier in reversed(pair_tiers):
                if nominal_value >= tier['minNotional']:
                    return (tier['maintenanceMarginRate'], tier['maintAmt'])
            raise ExchangeError('nominal value can not be lower than 0')
        else:
            raise ExchangeError(f'Cannot get maintenance ratio using {self.name}')