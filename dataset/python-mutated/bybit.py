""" Bybit exchange subclass """
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import ccxt
from freqtrade.constants import BuySell
from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, ExchangeError, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.util.datetime_helpers import dt_now, dt_ts
logger = logging.getLogger(__name__)

class Bybit(Exchange):
    """
    Bybit exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """
    _ft_has: Dict = {'ohlcv_candle_limit': 1000, 'ohlcv_has_history': True}
    _ft_has_futures: Dict = {'ohlcv_has_history': True, 'mark_ohlcv_timeframe': '4h', 'funding_fee_timeframe': '8h', 'stoploss_on_exchange': True, 'stoploss_order_types': {'limit': 'limit', 'market': 'market'}, 'stop_price_prop': 'stopPrice', 'stop_price_type_field': 'triggerBy', 'stop_price_type_value_mapping': {PriceType.LAST: 'LastPrice', PriceType.MARK: 'MarkPrice', PriceType.INDEX: 'IndexPrice'}}
    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [(TradingMode.FUTURES, MarginMode.ISOLATED)]

    @property
    def _ccxt_config(self) -> Dict:
        if False:
            print('Hello World!')
        config = {}
        if self.trading_mode == TradingMode.SPOT:
            config.update({'options': {'defaultType': 'spot'}})
        config.update(super()._ccxt_config)
        return config

    def market_is_future(self, market: Dict[str, Any]) -> bool:
        if False:
            print('Hello World!')
        main = super().market_is_future(market)
        return main and market['settle'] == 'USDT'

    @retrier
    def additional_exchange_init(self) -> None:
        if False:
            return 10
        '\n        Additional exchange initialization logic.\n        .api will be available at this point.\n        Must be overridden in child methods if required.\n        '
        try:
            if self.trading_mode == TradingMode.FUTURES and (not self._config['dry_run']):
                position_mode = self._api.set_position_mode(False)
                self._log_exchange_response('set_position_mode', position_mode)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def ohlcv_candle_limit(self, timeframe: str, candle_type: CandleType, since_ms: Optional[int]=None) -> int:
        if False:
            print('Hello World!')
        if candle_type in CandleType.FUNDING_RATE:
            return 200
        return super().ohlcv_candle_limit(timeframe, candle_type, since_ms)

    def _lev_prep(self, pair: str, leverage: float, side: BuySell, accept_fail: bool=False):
        if False:
            return 10
        if self.trading_mode != TradingMode.SPOT:
            params = {'leverage': leverage}
            self.set_margin_mode(pair, self.margin_mode, accept_fail=True, params=params)
            self._set_leverage(leverage, pair, accept_fail=True)

    def _get_params(self, side: BuySell, ordertype: str, leverage: float, reduceOnly: bool, time_in_force: str='GTC') -> Dict:
        if False:
            while True:
                i = 10
        params = super()._get_params(side=side, ordertype=ordertype, leverage=leverage, reduceOnly=reduceOnly, time_in_force=time_in_force)
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            params['position_idx'] = 0
        return params

    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, mm_ex_1: float=0.0, upnl_ex_1: float=0.0) -> Optional[float]:
        if False:
            print('Hello World!')
        '\n        Important: Must be fetching data from cached values as this is used by backtesting!\n        PERPETUAL:\n         bybit:\n          https://www.bybithelp.com/HelpCenterKnowledge/bybitHC_Article?language=en_US&id=000001067\n\n        Long:\n        Liquidation Price = (\n            Entry Price * (1 - Initial Margin Rate + Maintenance Margin Rate)\n            - Extra Margin Added/ Contract)\n        Short:\n        Liquidation Price = (\n            Entry Price * (1 + Initial Margin Rate - Maintenance Margin Rate)\n            + Extra Margin Added/ Contract)\n\n        Implementation Note: Extra margin is currently not used.\n\n        :param pair: Pair to calculate liquidation price for\n        :param open_rate: Entry price of position\n        :param is_short: True if the trade is a short, false otherwise\n        :param amount: Absolute value of position size incl. leverage (in base currency)\n        :param stake_amount: Stake amount - Collateral in settle currency.\n        :param leverage: Leverage used for this position.\n        :param trading_mode: SPOT, MARGIN, FUTURES, etc.\n        :param margin_mode: Either ISOLATED or CROSS\n        :param wallet_balance: Amount of margin_mode in the wallet being used to trade\n            Cross-Margin Mode: crossWalletBalance\n            Isolated-Margin Mode: isolatedWalletBalance\n        '
        market = self.markets[pair]
        (mm_ratio, _) = self.get_maintenance_ratio_and_amt(pair, stake_amount)
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            if market['inverse']:
                raise OperationalException('Freqtrade does not yet support inverse contracts')
            initial_margin_rate = 1 / leverage
            if is_short:
                return open_rate * (1 + initial_margin_rate - mm_ratio)
            else:
                return open_rate * (1 - initial_margin_rate + mm_ratio)
        else:
            raise OperationalException('Freqtrade only supports isolated futures for leverage trading')

    def get_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime) -> float:
        if False:
            print('Hello World!')
        '\n        Fetch funding fees, either from the exchange (live) or calculates them\n        based on funding rate/mark price history\n        :param pair: The quote/base pair of the trade\n        :param is_short: trade direction\n        :param amount: Trade amount\n        :param open_date: Open date of the trade\n        :return: funding fee since open_date\n        :raises: ExchangeError if something goes wrong.\n        '
        if self.trading_mode == TradingMode.FUTURES:
            try:
                return self._fetch_and_calculate_funding_fees(pair, amount, is_short, open_date)
            except ExchangeError:
                logger.warning(f'Could not update funding fees for {pair}.')
        return 0.0

    def fetch_orders(self, pair: str, since: datetime, params: Optional[Dict]=None) -> List[Dict]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch all orders for a pair "since"\n        :param pair: Pair for the query\n        :param since: Starting time for the query\n        '
        orders = []
        while since < dt_now():
            until = since + timedelta(days=7, minutes=-1)
            orders += super().fetch_orders(pair, since, params={'until': dt_ts(until)})
            since = until
        return orders

    def fetch_order(self, order_id: str, pair: str, params: Dict={}) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        order = super().fetch_order(order_id, pair, params)
        if order.get('status') == 'canceled' and order.get('filled') == 0.0 and (order.get('remaining') == 0.0):
            order['remaining'] = None
        return order